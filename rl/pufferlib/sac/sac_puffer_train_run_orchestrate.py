from __future__ import annotations

import contextlib
import importlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .types import SACConfig, TrainResult

_sac = "rl.pufferlib.sac"


def _env_utils():
    return importlib.import_module(f"{_sac}.env_utils")


def _eval_utils():
    return importlib.import_module(f"{_sac}.eval_utils")


def _model_utils():
    return importlib.import_module(f"{_sac}.model_utils")


def _loop_impl():
    return importlib.import_module(f"{_sac}.sac_loop_impl")


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
    actor_state_mod = importlib.import_module("rl.core.actor_state")
    checkpointing_mod = importlib.import_module("rl.checkpointing")
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
    _loop_impl()._restore_if_requested(
        config,
        modules,
        optimizers,
        replay,
        state,
        device=device,
        load_checkpoint=checkpointing_mod.load_checkpoint,
        restore_backbone_head_snapshot=actor_state_mod.restore_backbone_head_snapshot,
        restore_rng_state_payload=actor_state_mod.restore_rng_state_payload,
    )
    return (modules, optimizers, replay, state)


def train_sac_puffer_impl(config: SACConfig) -> TrainResult:
    eval_noise = importlib.import_module("rl.eval_noise")
    experiment_seeds = importlib.import_module("common.experiment_seeds")
    envu = _env_utils()
    ev = _eval_utils()
    mu = _model_utils()
    loop_impl = _loop_impl()
    engine_utils = importlib.import_module("rl.pufferlib.offpolicy.engine_utils")
    update_chunks = importlib.import_module("rl.core.update_chunks")
    actor_state_mod = importlib.import_module("rl.core.actor_state")
    eval_noise.normalize_eval_noise_mode(config.eval_noise_mode)
    _exp_dir, metrics_path, checkpoint_manager = _train_run_init_artifacts(config)
    env_setup, device = _train_run_init_runtime(config)
    if not isinstance(device, torch.device):
        raise TypeError(device)
    config.env_conf = env_setup.env_conf
    config.problem_seed = int(env_setup.problem_seed)
    noise_seed_0 = getattr(env_setup, "noise_seed_0", config.noise_seed_0)
    if noise_seed_0 is None:
        noise_seed_0 = experiment_seeds.resolve_noise_seed_0(problem_seed=int(config.problem_seed), noise_seed_0=None)
    config.noise_seed_0 = int(noise_seed_0)
    with contextlib.closing(envu.make_vector_env(config)) as envs:
        obs_np, _ = envs.reset(seed=int(env_setup.problem_seed))
        obs_spec = envu.infer_observation_spec(config, obs_np)
        obs_batch = envu.prepare_obs_np(obs_np, obs_spec=obs_spec)
        modules, optimizers, replay, state = _train_run_build_components(config, env_setup, obs_spec, obs_batch, device=device)
        _train_run_log_header(config, obs_batch, env_setup, obs_spec, device=device)
        loop_impl._train_loop(
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
            prepare_obs_np=envu.prepare_obs_np,
            to_env_action=envu.to_env_action,
            append_eval_metric=ev.append_eval_metric,
            log_if_due=ev.log_if_due,
            maybe_eval=ev.maybe_eval,
            sac_update=mu.sac_update,
            run_chunked_updates=update_chunks.run_chunked_updates,
            due_mark_fn=ev.due_mark,
            checkpoint_mark_if_due=engine_utils.checkpoint_mark_if_due,
            capture_backbone_head_snapshot=actor_state_mod.capture_backbone_head_snapshot,
        )
        if int(config.checkpoint_interval_steps or 0) > 0:
            checkpoint_manager.save_both(
                loop_impl._state_payload(
                    modules,
                    optimizers,
                    replay,
                    state,
                    capture_backbone_head_snapshot=actor_state_mod.capture_backbone_head_snapshot,
                ),
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
