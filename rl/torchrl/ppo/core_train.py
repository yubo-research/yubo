from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim

from rl import checkpointing
from rl.core import episode_rollout, ppo_eval, ppo_metrics, ppo_reward_norm
from rl.core.profiler import run_with_profiler
from rl.core.rollout_metrics import update_onpolicy_rollout_metrics
from rl.eval_noise import build_eval_plan
from rl.iter_record import IterInputs

from . import actor_eval
from .checkpoint_io import save_periodic_checkpoint
from .config import PPOConfig
from .core_env_setup import _build_eval_env_conf
from .core_types import _EnvSetup, _Modules, _TrainingSetup, _TrainState
from .core_utils import _is_due, _resolve_observation_contract_for_env


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
    state.reward_return = loaded.get("reward_return")
    state.reward_var = loaded.get("reward_var")
    state.reward_count = float(loaded.get("reward_count", state.reward_count))
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


def _empty_ppo_update_stats() -> dict[str, list[float]]:
    return {
        "approx_kls": [],
        "clipfracs": [],
        "ess_values": [],
        "skipped_updates": [],
        "loss_objective": [],
        "loss_critic": [],
        "loss_entropy": [],
        "grad_norm": [],
    }


def _append_optional_loss_stats(loss_td, update_stats: dict[str, list[float]]) -> None:
    if "clip_fraction" in loss_td.keys():
        update_stats["clipfracs"].append(float(loss_td["clip_fraction"]))
    if "kl_approx" in loss_td.keys():
        update_stats["approx_kls"].append(float(loss_td["kl_approx"]))
    if "ESS" in loss_td.keys():
        update_stats["ess_values"].append(float(loss_td["ESS"]))


def _run_ppo_minibatch_update(
    config: PPOConfig,
    training: _TrainingSetup,
    mb,
    update_stats: dict[str, list[float]],
) -> None:
    loss_td = training.loss_module(mb)
    loss_objective = loss_td["loss_objective"]
    loss_critic = loss_td["loss_critic"]
    loss_entropy = loss_td["loss_entropy"]
    loss = loss_objective + loss_critic + loss_entropy
    training.optimizer.zero_grad(set_to_none=True)
    if not bool(torch.isfinite(loss).all()):
        update_stats["skipped_updates"].append(1.0)
        return
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(training.train_params, config.optim.max_grad_norm)
    if not bool(torch.isfinite(grad_norm).all()):
        training.optimizer.zero_grad(set_to_none=True)
        update_stats["skipped_updates"].append(1.0)
        return
    training.optimizer.step()
    update_stats["skipped_updates"].append(0.0)
    update_stats["loss_objective"].append(float(loss_objective.detach()))
    update_stats["loss_critic"].append(float(loss_critic.detach()))
    update_stats["loss_entropy"].append(float(loss_entropy.detach()))
    update_stats["grad_norm"].append(float(grad_norm.detach()))
    _append_optional_loss_stats(loss_td, update_stats)


def _ppo_update(
    config: PPOConfig,
    training: _TrainingSetup,
    state: _TrainState,
    *,
    batch,
    device: torch.device,
) -> dict[str, list[float]]:
    batch = batch.to(device)
    _replace_nonfinite_rewards(batch)
    ppo_reward_norm.normalize_rewards_for_training(config, state, batch, device=device)
    training.gae(batch)
    nonfinite_gae_fraction = _replace_nonfinite_gae_targets(batch)
    flat = batch.reshape(-1)
    batch_size = int(flat.batch_size[0])
    minibatch_size = int(config.optim.minibatch_size)
    if minibatch_size <= 0:
        raise ValueError("optim.minibatch_size must be > 0.")
    if minibatch_size > batch_size:
        raise ValueError("optim.minibatch_size too large for batch_size.")
    update_stats = _empty_ppo_update_stats()
    for _ in range(int(config.optim.num_epochs)):
        indices = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, minibatch_size):
            mb_idx = indices[start : start + minibatch_size]
            _run_ppo_minibatch_update(config, training, flat[mb_idx], update_stats)
        mean_kl = ppo_metrics.finite_mean(update_stats["approx_kls"])
        if config.loss.target_kl is not None and mean_kl is not None and (mean_kl > float(config.loss.target_kl)):
            break
    update_stats["nonfinite_gae_fractions"] = [nonfinite_gae_fraction]
    return update_stats


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
        action_low=env.action_low,
        action_high=env.action_high,
    )
    traj, _ = episode_rollout.collect_denoised_trajectory(
        eval_env,
        eval_policy,
        num_denoise=config.eval.num_denoise,
        i_noise=int(eval_seed),
    )
    return float(traj.rreturn)


def _maybe_eval_and_log(
    config: PPOConfig,
    env: _EnvSetup,
    modules: _Modules,
    training: _TrainingSetup,
    state: _TrainState,
    *,
    iteration: int,
    iter_dt: float = 0.0,
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
    algo_metrics = ppo_metrics.build_algo_metrics(
        approx_kls=update_stats.get("approx_kls", []),
        clipfracs=update_stats.get("clipfracs", []),
    )
    from rl import logger
    from rl.torchrl_metrics import build_ppo_iter_record

    rollout_metrics = rollout_metrics or {}
    rollout_return = rollout_metrics.get("rollout_return")
    if not do_eval:
        _update_best_from_eval_return(state, modules, rollout_return)
        if do_log:
            elapsed = time.time() - start_time
            record = build_ppo_iter_record(
                _ppo_iter_inputs(
                    iteration,
                    training,
                    elapsed,
                    iter_dt,
                    algo_metrics,
                    rollout_metrics,
                    state,
                    rollout_return=rollout_return,
                )
            )
            ppo_metrics.enrich_ppo_iter_record(
                record,
                rollout_metrics=rollout_metrics,
                update_stats=update_stats,
            )
            ppo_metrics.log_record_diagnostics(record, iteration=iteration)
            logger.log_rl_iter(record, metrics_path=training.metrics_path)
        return
    plan = build_eval_plan(
        current=iteration,
        interval=int(config.eval.interval),
        seed=int(env.problem_seed),
        eval_seed_base=config.eval.seed_base,
        eval_noise_mode=config.eval.noise_mode,
    )
    eval_start = time.time()
    state.last_eval_return = _evaluate_actor(config, env, modules, device=device, eval_seed=plan.eval_seed)
    eval_dt = time.time() - eval_start
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
            action_low=getattr(env, "action_low", None),
            action_high=getattr(env, "action_high", None),
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
    if do_log or do_eval:
        record = build_ppo_iter_record(
            _ppo_iter_inputs(
                iteration,
                training,
                elapsed,
                iter_dt,
                algo_metrics,
                rollout_metrics,
                state,
                eval_return=float(state.last_eval_return),
                eval_dt=float(eval_dt),
            )
        )
        ppo_metrics.enrich_ppo_iter_record(
            record,
            rollout_metrics=rollout_metrics,
            update_stats=update_stats,
        )
        ppo_metrics.log_record_diagnostics(record, iteration=iteration)
        logger.log_rl_iter(record, metrics_path=training.metrics_path)


def _ppo_iter_inputs(
    iteration: int,
    training: _TrainingSetup,
    elapsed: float,
    iter_dt: float,
    algo_metrics: dict[str, float | None],
    rollout_metrics: dict[str, float | None],
    state: _TrainState,
    *,
    rollout_return: float | None = None,
    eval_return: float | None = None,
    eval_dt: float | None = None,
) -> IterInputs:
    frames = int(training.frames_per_batch)
    return IterInputs(
        iteration=int(iteration),
        step=int(iteration) * frames,
        frames_per_iter=frames,
        elapsed=float(elapsed),
        iter_dt=float(iter_dt),
        metrics={
            "kl": algo_metrics["kl"],
            "clipfrac": algo_metrics["clipfrac"],
            "rew": rollout_metrics.get("rollout_reward"),
            "ret_rollout": rollout_return if rollout_return is not None else rollout_metrics.get("rollout_return"),
            "ep_len": rollout_metrics.get("rollout_length"),
            "ret_best": float(state.best_return),
            "ret_eval": eval_return,
            "ret_heldout": state.last_heldout_return,
            "eval_dt": eval_dt,
        },
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
        iter_start = time.time()
        rollout_metrics = update_onpolicy_rollout_metrics(state, batch, num_envs=int(config.collector.num_envs))
        _anneal_lr(
            config,
            training.optimizer,
            iteration=iteration,
            num_iterations=training.num_iterations,
        )
        update_stats = _ppo_update(
            config,
            training,
            state,
            batch=batch,
            device=device,
        )
        _maybe_eval_and_log(
            config,
            env,
            modules,
            training,
            state,
            iteration=iteration,
            iter_dt=time.time() - iter_start,
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
