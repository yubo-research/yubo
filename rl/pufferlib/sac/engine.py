from __future__ import annotations

import importlib
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl.core.actor_state import capture_backbone_head_snapshot, restore_backbone_head_snapshot, restore_rng_state_payload
from rl.core.update_chunks import run_chunked_updates

from .config import SACConfig, TrainResult
from .env_utils import build_env_setup, infer_observation_spec, make_vector_env, prepare_obs_np, resolve_device, seed_everything, to_env_action
from .eval_utils import TrainState, append_eval_metric, due_mark, log_if_due, maybe_eval, render_videos_if_enabled
from .model_utils import build_modules, build_optimizers, sac_update, use_actor_state
from .replay import make_replay_buffer, resolve_replay_backend

__all__ = ["SACConfig", "TrainResult", "register", "train_sac_puffer", "train_sac_puffer_impl"]


def _log_header(config: SACConfig, obs_batch: np.ndarray, env_setup, obs_spec, *, device: torch.device) -> None:
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
    )


def _random_actions(num_envs: int, act_dim: int) -> np.ndarray:
    return np.random.uniform(low=-1.0, high=1.0, size=(int(num_envs), int(act_dim))).astype(np.float32)


def _init_run_artifacts(config: SACConfig) -> tuple[Path, Path, Any]:
    write_config = importlib.import_module("analysis.data_io").write_config
    checkpoint_manager_cls = importlib.import_module("rl.checkpointing").CheckpointManager
    exp_dir = Path(config.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_config(str(exp_dir), config.to_dict())
    metrics_path = exp_dir / "metrics.jsonl"
    return (exp_dir, metrics_path, checkpoint_manager_cls(exp_dir=exp_dir))


def _init_runtime(config: SACConfig):
    global_seed_for_run = importlib.import_module("rl.core.env_conf").global_seed_for_run
    env_setup = build_env_setup(config)
    run_seed = global_seed_for_run(int(env_setup.problem_seed))
    seed_everything(int(run_seed))
    device = resolve_device(str(config.device))
    return (env_setup, device)


def _build_training_components(config: SACConfig, env_setup, obs_spec, obs_batch: np.ndarray, *, device: torch.device):
    modules = build_modules(config, env_setup, obs_spec, device=device)
    optimizers = build_optimizers(config, modules)
    replay_backend = resolve_replay_backend(str(config.replay_backend), device=device)
    replay = make_replay_buffer(
        obs_shape=tuple((int(v) for v in obs_batch.shape[1:])), act_dim=int(env_setup.act_dim), capacity=int(config.replay_size), backend=replay_backend
    )
    state = TrainState(start_time=float(time.time()))
    _restore_if_requested(config, modules, optimizers, replay, state, device=device)
    return (modules, optimizers, replay, state)


def _state_payload(modules, optimizers, replay, state: Any) -> dict[str, Any]:
    actor_snapshot = capture_backbone_head_snapshot(modules.actor_backbone, modules.actor_head, log_std=None, state_to_cpu=False)
    return {
        "step": int(state.global_step),
        "total_updates": int(state.total_updates),
        "actor_backbone": actor_snapshot["backbone"],
        "actor_head": actor_snapshot["head"],
        "q1": modules.q1.state_dict(),
        "q2": modules.q2.state_dict(),
        "q1_target": modules.q1_target.state_dict(),
        "q2_target": modules.q2_target.state_dict(),
        "obs_scaler": modules.obs_scaler.state_dict(),
        "log_alpha": modules.log_alpha.detach().cpu(),
        "actor_optimizer": optimizers.actor_optimizer.state_dict(),
        "critic_optimizer": optimizers.critic_optimizer.state_dict(),
        "alpha_optimizer": optimizers.alpha_optimizer.state_dict(),
        "best_return": float(state.best_return),
        "best_actor_state": state.best_actor_state,
        "last_eval_return": float(state.last_eval_return),
        "last_heldout_return": state.last_heldout_return,
        "replay_state": replay.state_dict(),
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_if_requested(config: SACConfig, modules, optimizers, replay, state: Any, *, device: torch.device) -> None:
    if not config.resume_from:
        return
    load_checkpoint = importlib.import_module("rl.checkpointing").load_checkpoint
    loaded = load_checkpoint(Path(config.resume_from), device=device)
    restore_backbone_head_snapshot(
        modules.actor_backbone, modules.actor_head, {"backbone": loaded["actor_backbone"], "head": loaded["actor_head"]}, log_std=None, device=device
    )
    modules.q1.load_state_dict(loaded["q1"])
    modules.q2.load_state_dict(loaded["q2"])
    modules.q1_target.load_state_dict(loaded["q1_target"])
    modules.q2_target.load_state_dict(loaded["q2_target"])
    if loaded.get("obs_scaler") is not None:
        modules.obs_scaler.load_state_dict(loaded["obs_scaler"])
    if loaded.get("log_alpha") is not None:
        modules.log_alpha.data.copy_(loaded["log_alpha"].to(device=device, dtype=modules.log_alpha.dtype))
    optimizers.actor_optimizer.load_state_dict(loaded["actor_optimizer"])
    optimizers.critic_optimizer.load_state_dict(loaded["critic_optimizer"])
    optimizers.alpha_optimizer.load_state_dict(loaded["alpha_optimizer"])
    if loaded.get("replay_state") is not None:
        replay.load_state_dict(loaded["replay_state"])
    state.global_step = int(loaded.get("step", state.global_step))
    state.total_updates = int(loaded.get("total_updates", state.total_updates))
    state.best_return = float(loaded.get("best_return", state.best_return))
    state.best_actor_state = loaded.get("best_actor_state")
    state.last_eval_return = float(loaded.get("last_eval_return", state.last_eval_return))
    state.last_heldout_return = loaded.get("last_heldout_return", state.last_heldout_return)
    restore_rng_state_payload(loaded)


def _checkpoint_if_due(config: SACConfig, checkpoint_manager: Any, modules, optimizers, replay, state: Any) -> None:
    mark = due_mark(state.global_step, config.checkpoint_interval_steps, state.ckpt_mark)
    if mark is None:
        return
    state.ckpt_mark = int(mark)
    checkpoint_manager.save_both(_state_payload(modules, optimizers, replay, state), iteration=int(state.global_step))


def _train_loop(
    config: SACConfig,
    env_setup,
    modules,
    optimizers,
    replay: Any,
    state: Any,
    obs_spec,
    obs_batch: np.ndarray,
    envs,
    *,
    device: torch.device,
    metrics_path: Path,
    checkpoint_manager: Any,
) -> None:
    total_steps = int(config.total_timesteps)
    num_envs = int(obs_batch.shape[0])
    while state.global_step < total_steps:
        if state.global_step < int(config.learning_starts):
            action_norm = _random_actions(num_envs, int(env_setup.act_dim))
        else:
            obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_t, _ = modules.actor.sample(obs_t, deterministic=False)
            action_norm = np.asarray(action_t.detach().cpu().numpy(), dtype=np.float32)
        action_env = to_env_action(action_norm, low=env_setup.action_low, high=env_setup.action_high)
        nxt_obs_np, reward_np, terminated_np, truncated_np, _ = envs.step(action_env)
        done_np = np.logical_or(terminated_np, truncated_np)
        nxt_obs = prepare_obs_np(nxt_obs_np, obs_spec=obs_spec)
        replay.add_batch(obs_batch, action_norm, reward_np, nxt_obs, done_np)
        obs_batch = nxt_obs
        state.global_step = int(min(total_steps, state.global_step + num_envs))
        should_update = state.global_step >= int(config.learning_starts) and state.global_step % int(max(1, config.update_every)) < num_envs
        if should_update and replay.size >= int(config.batch_size):

            def _run_one_update() -> None:
                actor_loss, critic_loss, alpha_loss = sac_update(config, modules, optimizers, replay, device=device)
                state.total_updates += 1
                state.last_loss_actor = float(actor_loss)
                state.last_loss_critic = float(critic_loss)
                state.last_loss_alpha = float(alpha_loss)

            run_chunked_updates(int(max(1, config.updates_per_step)), int(config.learner_update_chunk_size), _run_one_update)
        prev_eval_mark = int(state.eval_mark)
        maybe_eval(config, env_setup, modules, obs_spec, state, device=device)
        if int(state.eval_mark) != prev_eval_mark:
            append_eval_metric(metrics_path, state, step=int(state.global_step))
        log_if_due(config, state, step=int(state.global_step), frames_per_batch=int(max(1, num_envs)))
        _checkpoint_if_due(config, checkpoint_manager, modules, optimizers, replay, state)


def train_sac_puffer_impl(config: SACConfig) -> TrainResult:
    normalize_eval_noise_mode = importlib.import_module("rl.eval_noise").normalize_eval_noise_mode
    rl_logger = importlib.import_module("rl.logger")
    normalize_eval_noise_mode(config.eval_noise_mode)
    _exp_dir, metrics_path, checkpoint_manager = _init_run_artifacts(config)
    env_setup, device = _init_runtime(config)
    with closing(make_vector_env(config)) as envs:
        obs_np, _ = envs.reset(seed=int(config.seed))
        obs_spec = infer_observation_spec(config, obs_np)
        obs_batch = prepare_obs_np(obs_np, obs_spec=obs_spec)
        modules, optimizers, replay, state = _build_training_components(config, env_setup, obs_spec, obs_batch, device=device)
        _log_header(config, obs_batch, env_setup, obs_spec, device=device)
        _train_loop(
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
            checkpoint_manager.save_both(_state_payload(modules, optimizers, replay, state), iteration=int(state.global_step))
        if state.best_return == -float("inf"):
            state.best_return = float("nan")
        rl_logger.log_run_footer(
            best_return=float(state.best_return),
            total_iters_or_steps=int(state.global_step),
            total_time=float(time.time() - state.start_time),
            algo_name="sac",
            step_label="steps",
        )
        if state.best_actor_state is not None:
            with use_actor_state(modules, state.best_actor_state):
                render_videos_if_enabled(config, env_setup, modules, obs_spec, device=device)
        else:
            render_videos_if_enabled(config, env_setup, modules, obs_spec, device=device)
    return TrainResult(
        best_return=float(state.best_return),
        last_eval_return=float(state.last_eval_return),
        last_heldout_return=None if state.last_heldout_return is None else float(state.last_heldout_return),
        num_steps=int(state.global_step),
    )


def train_sac_puffer(config: SACConfig) -> TrainResult:
    return train_sac_puffer_impl(config)


def register() -> None:
    from rl.registry import register_algo

    register_algo("sac", SACConfig, train_sac_puffer, backend="pufferlib")
