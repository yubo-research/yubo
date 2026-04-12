from __future__ import annotations

import contextlib
import importlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import sac_loop_impl as sac_loop_impl_mod
from .config import SACConfig, TrainResult

_sac = "rl.pufferlib.sac"


def _env_utils():
    return importlib.import_module(f"{_sac}.env_utils")


def _eval_utils():
    return importlib.import_module(f"{_sac}.eval_utils")


def _model_utils():
    return importlib.import_module(f"{_sac}.model_utils")


def _train_run_log_header(
    config: SACConfig,
    obs_batch: np.ndarray,
    env_setup,
    obs_spec,
    *,
    device,
) -> None:
    rl_logger = importlib.import_module("rl.logger")
    num_envs = int(obs_batch.shape[0])
    rl_logger.log_run_header_basic(
        algo_name="sac",
        env_tag=str(config.env_tag),
        seed=int(config.seed),
        backbone_name=str(config.backbone_name),
        from_pixels=obs_spec.mode == "pixels",
        obs_dim=int(obs_spec.vector_dim or np.prod(obs_spec.raw_shape)),
        act_dim=int(env_setup.act_dim),
        frames_per_batch=int(max(1, num_envs)),
        num_iterations=int(max(1, int(config.total_timesteps) // max(1, num_envs))),
        device_type=str(device.type),
        config_obj=config,
    )


def _train_run_init_artifacts(config: SACConfig) -> tuple[Path, Path, Any]:
    engine_utils = importlib.import_module("rl.pufferlib.offpolicy.engine_utils")
    return engine_utils.init_run_artifacts(
        exp_dir=str(config.exp_dir),
        config_dict=config.to_dict(),
    )


def _train_run_init_runtime(config: SACConfig):
    engine_utils = importlib.import_module("rl.pufferlib.offpolicy.engine_utils")
    envu = _env_utils()
    return engine_utils.init_runtime(
        config,
        build_env_setup_fn=envu.build_env_setup,
        seed_everything_fn=envu.seed_everything,
        resolve_device_fn=envu.resolve_device,
    )


def _train_run_build_components(
    config: SACConfig,
    env_setup,
    obs_spec,
    obs_batch: np.ndarray,
    *,
    device,
):
    mu = _model_utils()
    replay_mod = importlib.import_module(f"{_sac}.replay")
    ev = _eval_utils()
    modules = mu.build_modules(config, env_setup, obs_spec, device=device)
    optimizers = mu.build_optimizers(config, modules)
    replay_backend = replay_mod.resolve_replay_backend(str(config.replay_backend), device=device)
    replay = replay_mod.make_replay_buffer(
        obs_shape=tuple((int(v) for v in obs_batch.shape[1:])),
        act_dim=int(env_setup.act_dim),
        capacity=int(config.replay_size),
        backend=replay_backend,
    )
    state = ev.TrainState(start_time=float(time.time()))
    sac_loop_impl_mod._restore_if_requested(config, modules, optimizers, replay, state, device=device)
    return (modules, optimizers, replay, state)


def train_sac_puffer_impl(config: SACConfig) -> TrainResult:
    eval_noise = importlib.import_module("rl.eval_noise")
    core_env_conf = importlib.import_module("rl.core.env_conf")
    envu = _env_utils()
    eval_noise.normalize_eval_noise_mode(config.eval_noise_mode)
    _exp_dir, metrics_path, checkpoint_manager = _train_run_init_artifacts(config)
    env_setup, device = _train_run_init_runtime(config)
    if not isinstance(device, torch.device):
        raise TypeError(device)
    config.problem_seed = int(env_setup.problem_seed)
    noise_seed_0 = getattr(env_setup, "noise_seed_0", config.noise_seed_0)
    if noise_seed_0 is None:
        noise_seed_0 = core_env_conf.resolve_noise_seed_0(problem_seed=int(config.problem_seed), noise_seed_0=None)
    config.noise_seed_0 = int(noise_seed_0)
    with contextlib.closing(envu.make_vector_env(config)) as envs:
        obs_np, _ = envs.reset(seed=int(env_setup.problem_seed))
        obs_spec = envu.infer_observation_spec(config, obs_np)
        obs_batch = envu.prepare_obs_np(obs_np, obs_spec=obs_spec)
        modules, optimizers, replay, state = _train_run_build_components(config, env_setup, obs_spec, obs_batch, device=device)
        _train_run_log_header(config, obs_batch, env_setup, obs_spec, device=device)
        sac_loop_impl_mod._train_loop(
            config,
            env_setup,
            modules,
            optimizers,
            replay,
            state,
            obs_spec,
            obs_batch,
            envs,
            device=device,
            metrics_path=metrics_path,
            checkpoint_manager=checkpoint_manager,
        )
        if int(config.checkpoint_interval_steps or 0) > 0:
            checkpoint_manager.save_both(
                sac_loop_impl_mod._state_payload(modules, optimizers, replay, state),
                iteration=int(state.global_step),
            )
        if state.best_return == -float("inf"):
            state.best_return = float("nan")
        rl_logger = importlib.import_module("rl.logger")
        rl_logger.log_run_footer(
            best_return=float(state.best_return),
            total_iters_or_steps=int(state.global_step),
            total_time=float(time.time() - state.start_time),
            algo_name="sac",
            step_label="steps",
        )
        ev = _eval_utils()
        mu = _model_utils()
        if state.best_actor_state is not None:
            with mu.use_actor_state(modules, state.best_actor_state):
                ev.render_videos_if_enabled(config, env_setup, modules, obs_spec, device=device)
        else:
            ev.render_videos_if_enabled(config, env_setup, modules, obs_spec, device=device)
    return TrainResult(
        best_return=float(state.best_return),
        last_eval_return=float(state.last_eval_return),
        last_heldout_return=None if state.last_heldout_return is None else float(state.last_heldout_return),
        num_steps=int(state.global_step),
    )
