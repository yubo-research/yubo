from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim

from rl import checkpointing
from rl.core import episode_rollout, ppo_eval, ppo_metrics
from rl.core.profiler import run_with_profiler
from rl.eval_noise import build_eval_plan

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
    if not config.resume_from:
        return state
    resume_path = Path(config.resume_from)
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
    if not config.anneal_lr:
        return
    frac = 1.0 - (float(iteration) - 1.0) / float(num_iterations)
    lr_now = float(frac) * float(config.learning_rate)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_now


def _ppo_update(config: PPOConfig, training: _TrainingSetup, *, batch, device: torch.device) -> tuple[list[float], list[float]]:
    batch = batch.to(device)
    training.gae(batch)
    flat = batch.reshape(-1)
    batch_size = int(flat.batch_size[0])
    minibatch_size = int(batch_size // config.num_minibatches)
    if minibatch_size <= 0:
        raise ValueError("num_minibatches too large for batch_size.")
    clipfracs: list[float] = []
    approx_kls: list[float] = []
    for _ in range(int(config.update_epochs)):
        indices = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, minibatch_size):
            mb_idx = indices[start : start + minibatch_size]
            mb = flat[mb_idx]
            loss_td = training.loss_module(mb)
            loss = loss_td["loss_objective"] + loss_td["loss_critic"] + loss_td["loss_entropy"]
            training.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(training.train_params, config.max_grad_norm)
            training.optimizer.step()
            if "clip_fraction" in loss_td.keys():
                clipfracs.append(float(loss_td["clip_fraction"]))
            if "kl_approx" in loss_td.keys():
                approx_kls.append(float(loss_td["kl_approx"]))
        if config.target_kl is not None and approx_kls and (float(np.mean(approx_kls)) > float(config.target_kl)):
            break
    return (approx_kls, clipfracs)


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
    traj, _ = episode_rollout.collect_denoised_trajectory(eval_env, eval_policy, num_denoise=config.num_denoise, i_noise=int(eval_seed))
    return float(traj.rreturn)


def _maybe_eval_and_log(
    config: PPOConfig,
    env: _EnvSetup,
    modules: _Modules,
    training: _TrainingSetup,
    state: _TrainState,
    *,
    iteration: int,
    approx_kls: list[float],
    clipfracs: list[float],
    device: torch.device,
    start_time: float,
) -> None:
    do_eval = _is_due(int(iteration), int(config.eval_interval))
    do_log = _is_due(int(iteration), int(config.log_interval))
    if not do_eval:
        if do_log:
            from rl import logger

            elapsed = time.time() - start_time
            logger.log_progress_iteration(
                iteration,
                training.num_iterations,
                training.frames_per_batch,
                elapsed,
                algo_name="ppo",
            )
        return
    plan = build_eval_plan(
        current=iteration,
        interval=int(config.eval_interval),
        seed=int(env.problem_seed),
        eval_seed_base=config.eval_seed_base,
        eval_noise_mode=config.eval_noise_mode,
    )
    state.last_eval_return = _evaluate_actor(config, env, modules, device=device, eval_seed=plan.eval_seed)
    state.best_return, state.best_actor_state, _ = ppo_eval.update_best_actor_if_improved(
        eval_return=float(state.last_eval_return),
        best_return=float(state.best_return),
        best_actor_state=state.best_actor_state,
        capture_actor_state=lambda: actor_eval.capture_actor_snapshot(modules),
    )
    state.last_heldout_return = None
    if config.num_denoise_passive is not None and state.best_actor_state is not None:
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
            num_denoise_passive=config.num_denoise_passive,
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
        approx_kl=float(np.mean(approx_kls)) if approx_kls else None,
        clipfrac=float(np.mean(clipfracs)) if clipfracs else None,
        started_at=float(start_time),
        now=float(start_time + elapsed),
    )
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
            algo_metrics={"kl": record["approx_kl"], "clipfrac": record["clipfrac"]},
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
        _anneal_lr(
            config,
            training.optimizer,
            iteration=iteration,
            num_iterations=training.num_iterations,
        )
        approx_kls, clipfracs = _ppo_update(config, training, batch=batch, device=device)
        _maybe_eval_and_log(
            config,
            env,
            modules,
            training,
            state,
            iteration=iteration,
            approx_kls=approx_kls,
            clipfracs=clipfracs,
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

    if getattr(config, "profile_enable", False):
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
