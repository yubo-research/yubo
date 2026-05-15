"""PPO training entrypoints and re-exports (implementation lives in core_* modules)."""

from __future__ import annotations

import time

import torch

from common import experiment_seeds
from common.seed_all import seed_all
from rl import registry
from rl.core import env_contract, runtime
from rl.eval_noise import normalize_eval_noise_mode

from . import actor_eval, models
from .checkpoint_io import save_final_checkpoint
from .config import _PPO_RUNTIME_CAPABILITIES, PPOConfig, TrainResult
from .core_build import _build_collector, build_modules, build_training
from .core_env_setup import _make_video_context, build_env_setup
from .core_train import (
    _evaluate_actor,
    _maybe_eval_and_log,
    _resume_if_requested,
    _run_training_loop,
)
from .core_types import _TanhNormal, _TrainState

_ActorNet = models.ActorNet
_CriticNet = models.CriticNet
_DiscreteActorNet = models.DiscreteActorNet
_capture_actor_state = actor_eval.capture_actor_snapshot
_prepare_obs_for_backbone = models.prepare_obs_for_backbone
_restore_actor_state = actor_eval.restore_actor_snapshot

__all__ = [
    "PPOConfig",
    "TrainResult",
    "_ActorNet",
    "_CriticNet",
    "_DiscreteActorNet",
    "_TanhNormal",
    "_TrainState",
    "_capture_actor_state",
    "_evaluate_actor",
    "_maybe_eval_and_log",
    "_prepare_obs_for_backbone",
    "_restore_actor_state",
    "build_env_setup",
    "build_modules",
    "build_training",
    "register",
    "torch",
    "runtime",
    "train_ppo",
]


def _log_ppo_config(config, env, training, runtime, from_pixels, backbone_info):
    print(
        f"[rl/ppo/torchrl] env_tag={config.env_tag} exp_dir={training.exp_dir} seed={config.seed} problem_seed={env.problem_seed} device={runtime.device.type} obs_dim={env.obs_dim} act_dim={env.act_dim} total_timesteps={config.total_timesteps} num_envs={config.num_envs} num_steps={config.num_steps} frames_per_batch={training.frames_per_batch} num_iterations={training.num_iterations} update_epochs={config.update_epochs} eval_interval={config.eval_interval} collector={runtime.collector_backend} single_env={runtime.single_env_backend} from_pixels={from_pixels}{backbone_info} share_backbone={bool(config.share_backbone)}",
        flush=True,
    )
    print(
        f"[rl/ppo/torchrl] actor_head={list(config.actor_head_hidden_sizes)} value_head={list(config.critic_head_hidden_sizes)} log_std_init={config.log_std_init} lr={config.learning_rate} gamma={config.gamma} gae_lambda={config.gae_lambda} clip_coef={config.clip_coef} vf_coef={config.vf_coef} ent_coef={config.ent_coef}",
        flush=True,
    )


def train_ppo(config: PPOConfig) -> TrainResult:
    if config.eval_noise_mode is not None:
        normalize_eval_noise_mode(config.eval_noise_mode)
    resolved = experiment_seeds.resolve_run_seeds(
        seed=int(config.seed),
        problem_seed=config.problem_seed,
        noise_seed_0=config.noise_seed_0,
    )
    config.problem_seed = int(resolved.problem_seed)
    config.noise_seed_0 = int(resolved.noise_seed_0)
    seed_all(experiment_seeds.global_seed_for_run(int(resolved.problem_seed)))
    env = build_env_setup(config)
    runtime = config.resolve_runtime(capabilities=_PPO_RUNTIME_CAPABILITIES)
    modules = build_modules(config, env, device=runtime.device)
    training = build_training(config, env, modules, runtime=runtime)
    state = _resume_if_requested(config, modules, training, device=runtime.device)
    from_pixels = env.io_contract.observation.mode == "pixels"
    backbone_resolved = env_contract.resolve_backbone_name(config.backbone_name, env.io_contract.observation)
    is_cnn = backbone_resolved in {"nature_cnn", "nature_cnn_atari"}
    backbone_info = f" backbone={backbone_resolved}"
    if not is_cnn:
        backbone_info += f" hidden={list(config.backbone_hidden_sizes)} act={config.backbone_activation} ln={bool(config.backbone_layer_norm)}"
    _log_ppo_config(config, env, training, runtime, from_pixels, backbone_info)
    remaining_iterations = max(0, training.num_iterations - state.start_iteration)
    video, ctx = _make_video_context(config, env, from_pixels=from_pixels)
    if remaining_iterations == 0:
        video.render_policy_videos_rl(config, env, modules, training, state, ctx, device=runtime.device)
        return TrainResult(
            best_return=float(state.best_return),
            last_eval_return=float(state.last_eval_return),
            last_heldout_return=state.last_heldout_return,
            num_iterations=training.num_iterations,
        )
    collector = _build_collector(
        config,
        env,
        modules,
        training,
        runtime=runtime,
        remaining_iterations=remaining_iterations,
    )
    from rl import logger

    logger.log_run_header("ppo", config, env, training, runtime)
    train_start = time.time()
    try:
        _run_training_loop(
            config,
            env,
            modules,
            training,
            state,
            collector=collector,
            device=runtime.device,
        )
    finally:
        collector.shutdown()
    total_time = time.time() - train_start
    logger.log_run_footer(state.best_return, training.num_iterations, total_time, algo_name="ppo")
    save_final_checkpoint(config=config, training_setup=training, modules=modules, train_state=state)
    video.render_policy_videos_rl(config, env, modules, training, state, ctx, device=runtime.device)
    return TrainResult(
        best_return=float(state.best_return),
        last_eval_return=float(state.last_eval_return),
        last_heldout_return=state.last_heldout_return,
        num_iterations=training.num_iterations,
    )


def register():
    registry.register_algo("ppo", PPOConfig, train_ppo)
