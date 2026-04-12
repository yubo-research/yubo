from __future__ import annotations

import importlib
import time
from pathlib import Path

import numpy as np
import torch

from .config import PufferPPOConfig, TrainResult


def run_training(
    config: PufferPPOConfig,
    plan,
    device: torch.device,
    metrics_path: Path,
    envs,
    *,
    build_eval_env_conf_fn,
    init_runtime_fn,
    prepare_obs_fn,
) -> TrainResult:
    puffer_train_ops = importlib.import_module("rl.pufferlib.ppo.training_ops")
    puffer_ckpt = importlib.import_module("rl.pufferlib.ppo.checkpoint")
    puffer_metrics = importlib.import_module("rl.pufferlib.ppo.metrics")
    puffer_eval = importlib.import_module("rl.pufferlib.ppo.eval")
    checkpoint_manager_cls = importlib.import_module("rl.checkpointing").CheckpointManager
    rl_logger = importlib.import_module("rl.logger")
    model, optimizer, obs_shape, buffer, state = init_runtime_fn(config, plan, device, envs)
    checkpoint_manager = checkpoint_manager_cls(exp_dir=metrics_path.parent)
    puffer_ckpt.restore_checkpoint_if_requested(config, plan, model, optimizer, state, device=device)
    rl_logger.log_run_header_basic(
        algo_name="ppo",
        env_tag=str(config.env_tag),
        seed=int(config.seed),
        backbone_name=str(config.backbone_name),
        from_pixels=state.obs_spec.mode == "pixels",
        obs_dim=64 if state.obs_spec.mode == "pixels" else int(state.obs_spec.vector_dim or 1),
        act_dim=int(state.action_spec.dim),
        frames_per_batch=int(plan.batch_size),
        num_iterations=int(plan.num_iterations),
        device_type=str(device.type),
        config_obj=config,
    )
    b_inds = np.arange(plan.batch_size)
    for iteration in range(state.start_iteration + 1, plan.num_iterations + 1):
        puffer_metrics._maybe_anneal_lr(config, plan, optimizer, iteration)
        puffer_train_ops.collect_rollout(plan, model, envs, buffer, state, device, prepare_obs_fn=prepare_obs_fn)
        advantages, returns = puffer_train_ops.compute_advantages(plan, config, model, state, buffer, device)
        batch = puffer_train_ops.flatten_batch(plan, buffer, advantages, returns, obs_shape)
        update_stats = puffer_train_ops.ppo_update(config, plan, model, optimizer, batch, b_inds)
        puffer_eval.maybe_eval_and_update_state(
            config,
            model,
            state,
            iteration=iteration,
            device=device,
            build_eval_env_conf_fn=build_eval_env_conf_fn,
            prepare_obs_fn=prepare_obs_fn,
        )
        metric = puffer_metrics._metric_payload(iteration, plan, optimizer, state, update_stats, batch)
        puffer_metrics._append_metrics_line(metrics_path, metric)
        puffer_metrics._log_iteration(config, metric)
        puffer_ckpt.maybe_save_periodic_checkpoint(config, checkpoint_manager, model, optimizer, state, iteration=iteration)
    best_return = state.best_return
    if best_return == -float("inf"):
        best_return = float("nan")
    total_time = time.time() - state.start_time
    rl_logger.log_run_footer(float(best_return), int(plan.num_iterations), float(total_time), algo_name="ppo")
    final_iteration = int(max(state.start_iteration, plan.num_iterations))
    puffer_ckpt.save_final_checkpoint(config, checkpoint_manager, model, optimizer, state, iteration=final_iteration)
    puffer_eval.maybe_render_videos(
        config,
        model,
        state,
        exp_dir=metrics_path.parent,
        device=device,
        build_eval_env_conf_fn=build_eval_env_conf_fn,
        prepare_obs_fn=prepare_obs_fn,
    )
    return TrainResult(
        best_return=float(best_return),
        last_eval_return=float(state.last_eval_return),
        last_heldout_return=puffer_metrics._as_optional_finite(state.last_heldout_return),
        num_iterations=int(plan.num_iterations),
    )
