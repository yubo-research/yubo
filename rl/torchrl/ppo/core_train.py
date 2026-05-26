from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim

from rl import checkpointing
from rl.core import episode_rollout, ppo_eval, ppo_metrics
from rl.core.profiler import run_with_profiler
from rl.core.rollout_metrics import update_onpolicy_rollout_metrics
from rl.eval_noise import build_eval_plan

from . import actor_eval
from .checkpoint_io import save_periodic_checkpoint
from .config import PPOConfig
from .core_env_setup import _build_eval_env_conf
from .core_types import _EnvSetup, _Modules, _TrainingSetup, _TrainState
from .core_utils import _is_due, _resolve_observation_contract_for_env

_LOGGER = logging.getLogger(__name__)


def _resume_if_requested(
    config: PPOConfig,
    modules: _Modules,
    training: _TrainingSetup,
    *,
    device: torch.device,
) -> _TrainState:
    state = _TrainState()
    if not config.checkpoint.resume_from:
        return state
    resume_path = Path(config.checkpoint.resume_from)
    loaded = checkpointing.load_checkpoint(resume_path, device=device)
    actor_snapshot: dict[str, Any] = {
        "backbone": loaded["actor_backbone"],
        "head": loaded["actor_head"],
    }
    if "log_std" in loaded:
        actor_snapshot["log_std"] = loaded["log_std"]
    actor_eval.restore_actor_snapshot(modules, actor_snapshot, device=device)
    modules.critic_backbone.load_state_dict(loaded["critic_backbone"])
    modules.critic_head.load_state_dict(loaded["critic_head"])
    if loaded.get("obs_scaler") is not None:
        modules.obs_scaler.load_state_dict(loaded["obs_scaler"])
    training.optimizer.load_state_dict(loaded["optimizer"])
    state.start_iteration = int(loaded.get("iteration", 0))
    state.best_return = float(loaded.get("best_return", state.best_return))
    state.best_actor_state = loaded.get("best_actor_state")
    state.last_eval_return = float(loaded.get("last_eval_return", state.last_eval_return))
    state.last_heldout_return = loaded.get("last_heldout_return", state.last_heldout_return)
    if "rng_torch" in loaded:
        torch.set_rng_state(loaded["rng_torch"])
    if "rng_numpy" in loaded:
        np.random.set_state(loaded["rng_numpy"])
    if torch.cuda.is_available() and loaded.get("rng_cuda") is not None:
        torch.cuda.set_rng_state_all(loaded["rng_cuda"])
    return state


def _anneal_lr(
    config: PPOConfig,
    optimizer: optim.Optimizer,
    *,
    iteration: int,
    num_iterations: int,
) -> None:
    if not config.optim.anneal_lr:
        return
    frac = 1.0 - (float(iteration) - 1.0) / float(num_iterations)
    lr_now = float(frac) * float(config.optim.lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_now


def _ppo_update(config: PPOConfig, training: _TrainingSetup, *, batch, device: torch.device) -> dict[str, list[float]]:
    batch = batch.to(device)
    _replace_nonfinite_rewards(batch)
    training.gae(batch)
    nonfinite_gae_fraction = _replace_nonfinite_gae_targets(batch)
    flat = batch.reshape(-1)
    batch_size = int(flat.batch_size[0])
    minibatch_size = int(config.optim.minibatch_size)
    if minibatch_size <= 0:
        raise ValueError("optim.minibatch_size must be > 0.")
    if minibatch_size > batch_size:
        raise ValueError("optim.minibatch_size too large for batch_size.")
    clipfracs: list[float] = []
    approx_kls: list[float] = []
    ess_values: list[float] = []
    skipped_updates: list[float] = []
    loss_objective_values: list[float] = []
    loss_critic_values: list[float] = []
    loss_entropy_values: list[float] = []
    grad_norm_values: list[float] = []
    for _ in range(int(config.optim.num_epochs)):
        indices = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, minibatch_size):
            mb_idx = indices[start : start + minibatch_size]
            mb = flat[mb_idx]
            loss_td = training.loss_module(mb)
            loss_objective = loss_td["loss_objective"]
            loss_critic = loss_td["loss_critic"]
            loss_entropy = loss_td["loss_entropy"]
            loss = loss_objective + loss_critic + loss_entropy
            training.optimizer.zero_grad(set_to_none=True)
            if not bool(torch.isfinite(loss).all()):
                skipped_updates.append(1.0)
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(training.train_params, config.optim.max_grad_norm)
            if not bool(torch.isfinite(grad_norm).all()):
                training.optimizer.zero_grad(set_to_none=True)
                skipped_updates.append(1.0)
                continue
            training.optimizer.step()
            skipped_updates.append(0.0)
            loss_objective_values.append(float(loss_objective.detach()))
            loss_critic_values.append(float(loss_critic.detach()))
            loss_entropy_values.append(float(loss_entropy.detach()))
            grad_norm_values.append(float(grad_norm.detach()))
            if "clip_fraction" in loss_td.keys():
                clipfracs.append(float(loss_td["clip_fraction"]))
            if "kl_approx" in loss_td.keys():
                approx_kls.append(float(loss_td["kl_approx"]))
            if "ESS" in loss_td.keys():
                ess_values.append(float(loss_td["ESS"]))
        mean_kl = _finite_mean(approx_kls)
        if config.loss.target_kl is not None and mean_kl is not None and (mean_kl > float(config.loss.target_kl)):
            break
    return {
        "approx_kls": approx_kls,
        "clipfracs": clipfracs,
        "ess_values": ess_values,
        "nonfinite_gae_fractions": [nonfinite_gae_fraction],
        "skipped_updates": skipped_updates,
        "loss_objective": loss_objective_values,
        "loss_critic": loss_critic_values,
        "loss_entropy": loss_entropy_values,
        "grad_norm": grad_norm_values,
    }


def _replace_nonfinite_rewards(batch) -> None:
    try:
        rewards = batch[("next", "reward")]
    except Exception:
        return
    finite = torch.isfinite(rewards)
    if bool(finite.all()):
        return
    batch[("next", "reward")] = torch.where(finite, rewards, torch.zeros_like(rewards))


def _replace_nonfinite_gae_targets(batch) -> float:
    nonfinite_fractions: list[float] = []
    try:
        state_value = batch["state_value"].detach()
    except Exception:
        state_value = None
    for key in ("advantage", "value_target"):
        try:
            value = batch[key]
        except Exception:
            continue
        finite = torch.isfinite(value)
        nonfinite_fractions.append(float((~finite).float().mean()))
        if bool(finite.all()):
            continue
        replacement = torch.zeros_like(value)
        if key == "value_target" and state_value is not None and state_value.shape == value.shape:
            replacement = state_value
        batch[key] = torch.where(finite, value, replacement)
    return float(np.mean(nonfinite_fractions)) if nonfinite_fractions else 0.0


def _finite_mean(values: list[float]) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if int(finite.size) else None


def _algo_metrics(
    *,
    approx_kls: list[float],
    clipfracs: list[float],
) -> dict[str, float | None]:
    return {
        "kl": _finite_mean(approx_kls),
        "clipfrac": _finite_mean(clipfracs),
    }


def _update_record_diagnostics(
    record: dict,
    *,
    rollout_metrics: dict[str, float | None],
    update_stats: dict[str, list[float]],
) -> None:
    record["nonfinite_reward_fraction"] = rollout_metrics.get("nonfinite_reward_fraction")
    record["nonfinite_gae_fraction"] = _finite_mean(update_stats.get("nonfinite_gae_fractions", []))
    record["skipped_update_fraction"] = _finite_mean(update_stats.get("skipped_updates", []))
    record["ess"] = _finite_mean(update_stats.get("ess_values", []))
    record["loss_objective"] = _finite_mean(update_stats.get("loss_objective", []))
    record["loss_critic"] = _finite_mean(update_stats.get("loss_critic", []))
    record["loss_entropy"] = _finite_mean(update_stats.get("loss_entropy", []))
    record["grad_norm"] = _finite_mean(update_stats.get("grad_norm", []))


def _log_record_diagnostics(record: dict, *, iteration: int) -> None:
    nonfinite_reward = float(record.get("nonfinite_reward_fraction") or 0.0)
    nonfinite_gae = float(record.get("nonfinite_gae_fraction") or 0.0)
    skipped_update = float(record.get("skipped_update_fraction") or 0.0)
    ess = record.get("ess")
    if nonfinite_reward > 0.0 or skipped_update > 0.0:
        _LOGGER.warning(
            "ppo diagnostics iteration=%s nonfinite_reward_fraction=%.4g nonfinite_gae_fraction=%.4g skipped_update_fraction=%.4g ess=%s",
            int(iteration),
            nonfinite_reward,
            nonfinite_gae,
            skipped_update,
            "-" if ess is None else f"{float(ess):.4g}",
        )
        return
    if nonfinite_gae > 0.0:
        _LOGGER.debug(
            "ppo diagnostics iteration=%s nonfinite_gae_fraction=%.4g ess=%s",
            int(iteration),
            nonfinite_gae,
            "-" if ess is None else f"{float(ess):.4g}",
        )


def _update_best_from_eval_return(state: _TrainState, modules: _Modules, eval_return: float | None) -> None:
    if eval_return is None or not np.isfinite(float(eval_return)):
        return
    state.last_eval_return = float(eval_return)
    state.best_return, state.best_actor_state, _ = ppo_eval.update_best_actor_if_improved(
        eval_return=float(eval_return),
        best_return=float(state.best_return),
        best_actor_state=state.best_actor_state,
        capture_actor_state=lambda: actor_eval.capture_actor_snapshot(modules),
    )


def _evaluate_actor(
    config: PPOConfig,
    env: _EnvSetup,
    modules: _Modules,
    *,
    device: torch.device,
    eval_seed: int,
) -> float:
    obs_contract = _resolve_observation_contract_for_env(config, env)
    from_pixels = obs_contract.mode == "pixels"
    eval_env = _build_eval_env_conf(config, env, from_pixels=from_pixels)
    eval_policy = actor_eval.ActorEvalPolicy(
        modules.actor_backbone,
        modules.actor_head,
        modules.obs_scaler,
        device=device,
        obs_contract=obs_contract,
        is_discrete=bool(getattr(env, "is_discrete", False)),
    )
    traj, _ = episode_rollout.collect_denoised_trajectory(eval_env, eval_policy, num_denoise=config.eval.num_denoise, i_noise=int(eval_seed))
    return float(traj.rreturn)


def _maybe_eval_and_log(
    config: PPOConfig,
    env: _EnvSetup,
    modules: _Modules,
    training: _TrainingSetup,
    state: _TrainState,
    *,
    iteration: int,
    update_stats: dict[str, list[float]] | None = None,
    approx_kls: list[float] | None = None,
    clipfracs: list[float] | None = None,
    rollout_metrics: dict[str, float | None] | None = None,
    device: torch.device,
    start_time: float,
) -> None:
    update_stats = update_stats or {
        "approx_kls": approx_kls or [],
        "clipfracs": clipfracs or [],
    }
    do_eval = _is_due(int(iteration), int(config.eval.interval))
    do_log = _is_due(int(iteration), int(config.log_interval))
    algo_metrics = _algo_metrics(
        approx_kls=update_stats.get("approx_kls", []),
        clipfracs=update_stats.get("clipfracs", []),
    )
    rollout_metrics = rollout_metrics or {}
    rollout_return = rollout_metrics.get("rollout_return")
    if not do_eval:
        _update_best_from_eval_return(state, modules, rollout_return)
        if do_log:
            from rl import logger

            elapsed = time.time() - start_time
            global_step = iteration * training.frames_per_batch
            record = ppo_metrics.build_eval_record(
                iteration=int(iteration),
                global_step=int(global_step),
                eval_return=rollout_return,
                heldout_return=None,
                best_return=float(state.best_return),
                approx_kl=algo_metrics["kl"],
                clipfrac=algo_metrics["clipfrac"],
                rollout_reward=rollout_metrics.get("rollout_reward"),
                rollout_return=rollout_return,
                rollout_length=rollout_metrics.get("rollout_length"),
                started_at=float(start_time),
                now=float(start_time + elapsed),
            )
            _update_record_diagnostics(
                record,
                rollout_metrics=rollout_metrics,
                update_stats=update_stats,
            )
            _log_record_diagnostics(record, iteration=iteration)
            logger.append_metrics(training.metrics_path, record)
            logger.log_progress_iteration(
                iteration,
                training.num_iterations,
                training.frames_per_batch,
                elapsed,
                algo_metrics=algo_metrics,
                algo_name="ppo",
                eval_return=rollout_return,
                best_return=float(state.best_return),
            )
        return
    plan = build_eval_plan(
        current=iteration,
        interval=int(config.eval.interval),
        seed=int(env.problem_seed),
        eval_seed_base=config.eval.seed_base,
        eval_noise_mode=config.eval.noise_mode,
    )
    state.last_eval_return = _evaluate_actor(config, env, modules, device=device, eval_seed=plan.eval_seed)
    _update_best_from_eval_return(state, modules, state.last_eval_return)
    state.last_heldout_return = None
    if config.eval.num_denoise_passive is not None and state.best_actor_state is not None:
        obs_contract = _resolve_observation_contract_for_env(config, env)
        from_pixels = obs_contract.mode == "pixels"
        best_eval_policy = actor_eval.ActorEvalPolicy(
            modules.actor_backbone,
            modules.actor_head,
            modules.obs_scaler,
            device=device,
            obs_contract=obs_contract,
            is_discrete=bool(getattr(env, "is_discrete", False)),
        )
        state.last_heldout_return = ppo_eval.evaluate_heldout_with_best_actor(
            best_actor_state=state.best_actor_state,
            num_denoise_passive=config.eval.num_denoise_passive,
            heldout_i_noise=plan.heldout_i_noise,
            with_actor_state=lambda snapshot: actor_eval.use_actor_snapshot(modules, snapshot, device=device),
            evaluate_for_best=episode_rollout.evaluate_for_best,
            eval_env_conf=_build_eval_env_conf(config, env, from_pixels=from_pixels),
            eval_policy=best_eval_policy,
        )
    elapsed = time.time() - start_time
    global_step = iteration * training.frames_per_batch
    record = ppo_metrics.build_eval_record(
        iteration=int(iteration),
        global_step=int(global_step),
        eval_return=float(state.last_eval_return),
        heldout_return=state.last_heldout_return,
        best_return=float(state.best_return),
        approx_kl=algo_metrics["kl"],
        clipfrac=algo_metrics["clipfrac"],
        rollout_reward=rollout_metrics.get("rollout_reward"),
        rollout_return=rollout_metrics.get("rollout_return"),
        rollout_length=rollout_metrics.get("rollout_length"),
        started_at=float(start_time),
        now=float(start_time + elapsed),
    )
    _update_record_diagnostics(
        record,
        rollout_metrics=rollout_metrics,
        update_stats=update_stats,
    )
    _log_record_diagnostics(record, iteration=iteration)
    from rl import logger

    logger.append_metrics(training.metrics_path, record)
    if do_log:
        from rl import logger

        logger.log_eval_iteration(
            iteration,
            training.num_iterations,
            training.frames_per_batch,
            eval_return=state.last_eval_return,
            heldout_return=state.last_heldout_return,
            best_return=state.best_return,
            algo_metrics=algo_metrics,
            algo_name="ppo",
            elapsed=elapsed,
        )


def _run_training_loop(
    config: PPOConfig,
    env: _EnvSetup,
    modules: _Modules,
    training: _TrainingSetup,
    state: _TrainState,
    *,
    collector,
    device: torch.device,
) -> None:
    start_time = time.time()

    def run_iteration(iteration: int, batch):
        rollout_metrics = update_onpolicy_rollout_metrics(state, batch, num_envs=int(config.collector.num_envs))
        _anneal_lr(
            config,
            training.optimizer,
            iteration=iteration,
            num_iterations=training.num_iterations,
        )
        update_stats = _ppo_update(config, training, batch=batch, device=device)
        _maybe_eval_and_log(
            config,
            env,
            modules,
            training,
            state,
            iteration=iteration,
            update_stats=update_stats,
            rollout_metrics=rollout_metrics,
            device=device,
            start_time=start_time,
        )
        save_periodic_checkpoint(
            config=config,
            training_setup=training,
            modules=modules,
            train_state=state,
            iteration=iteration,
        )

    if config.profile.enable:
        run_with_profiler(
            config,
            collector,
            run_iteration,
            device=device,
            num_iterations=training.num_iterations,
            start_iteration=state.start_iteration,
        )
    else:
        for iteration, batch in enumerate(collector, start=state.start_iteration + 1):
            if iteration > training.num_iterations:
                break
            run_iteration(iteration, batch)
