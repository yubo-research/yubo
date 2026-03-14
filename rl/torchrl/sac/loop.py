from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from rl import logger as rl_logger
from rl.checkpointing import save_final_if_enabled, save_if_due
from rl.core import eval as rl_eval
from rl.core import sac_metrics
from rl.core.progress import is_due


def as_float32_observation(observation: np.ndarray) -> np.ndarray:
    return np.asarray(observation, dtype=np.float32)


def select_training_action(
    config: Any,
    env_setup: Any,
    modules: Any,
    *,
    step: int,
    observation: np.ndarray,
    train_env: Any,
    device: torch.device,
    unscale_action_from_env: Any,
    scale_action_to_env: Any,
) -> tuple[np.ndarray, np.ndarray]:
    obs_np = np.asarray(observation)
    if obs_np.dtype != np.float32:
        raise RuntimeError(
            f"Expected float32 observation from env pipeline, got {obs_np.dtype}. Ensure observations are normalized to float32 before policy calls."
        )
    if step < int(config.learning_starts):
        action_env = train_env.action_space.sample()
        action_norm = unscale_action_from_env(action_env, env_setup.action_low, env_setup.action_high)
        return (action_env, action_norm)
    with torch.no_grad():
        obs_td = TensorDict({"observation": torch.as_tensor(obs_np, device=device).unsqueeze(0)}, [1])
        action_norm = modules.actor(obs_td)["action"][0].detach().cpu().numpy()
    action_env = scale_action_to_env(action_norm, env_setup.action_low, env_setup.action_high)
    return (action_env, action_norm)


def advance_env_and_store(
    training_setup: Any,
    *,
    train_env: Any,
    observation: np.ndarray,
    action_env: np.ndarray,
    action_norm: np.ndarray,
    make_transition: Any,
) -> np.ndarray:
    next_observation, reward, terminated, truncated, _ = train_env.step(action_env)
    next_observation = as_float32_observation(next_observation)
    done = bool(terminated or truncated)
    training_setup.replay.add(
        make_transition(
            observation,
            action_norm,
            next_observation,
            float(reward),
            bool(terminated),
            done,
        )
    )
    if done:
        reset_observation, _ = train_env.reset()
        return as_float32_observation(reset_observation)
    return next_observation


def run_updates_if_due(
    config: Any,
    training_setup: Any,
    *,
    step: int,
    device: torch.device,
    latest_losses: dict[str, float],
    total_updates: int,
    update_step: Any,
) -> tuple[dict[str, float], int]:
    if step < int(config.learning_starts) or step % int(config.update_every) != 0:
        return (latest_losses, total_updates)
    for _ in range(int(config.updates_per_step)):
        latest_losses = update_step(training_setup, device=device, batch_size=int(config.batch_size))
        total_updates += 1
    return (latest_losses, total_updates)


@contextmanager
def temporary_actor_state(
    modules: Any,
    actor_state: dict,
    *,
    capture_actor_state: Any,
    restore_actor_state: Any,
):
    previous_actor_state = capture_actor_state(modules)
    restore_actor_state(modules, actor_state)
    try:
        yield
    finally:
        restore_actor_state(modules, previous_actor_state)


def heldout(
    config: Any,
    env_setup: Any,
    modules: Any,
    train_state: Any,
    *,
    device: torch.device,
    capture_actor_state: Any,
    restore_actor_state: Any,
    eval_policy_factory: Any,
    get_env_conf: Any,
    best: Any,
    heldout_i_noise: int = 99999,
) -> float | None:
    best_eval_policy = eval_policy_factory(modules, env_setup, device)
    env_conf = get_env_conf(
        config.env_tag,
        problem_seed=env_setup.problem_seed,
        noise_seed_0=env_setup.noise_seed_0,
        obs_mode=str(getattr(getattr(env_setup, "env_conf", None), "obs_mode", "state")),
    )
    return rl_eval.heldout(
        best_actor_state=train_state.best_actor_state,
        num_denoise_passive=config.num_denoise_passive,
        heldout_i_noise=int(heldout_i_noise),
        with_actor_state=lambda snapshot: temporary_actor_state(
            modules,
            snapshot,
            capture_actor_state=capture_actor_state,
            restore_actor_state=restore_actor_state,
        ),
        best=best,
        eval_env_conf=env_conf,
        eval_policy=best_eval_policy,
    )


def evaluate_if_due(
    config: Any,
    env_setup: Any,
    modules: Any,
    training_setup: Any,
    train_state: Any,
    *,
    step: int,
    device: torch.device,
    start_time: float,
    latest_losses: dict[str, float],
    total_updates: int,
    evaluate_actor: Any,
    capture_actor_state: Any,
    evaluate_heldout: Any,
) -> None:
    if not is_due(step, config.eval_interval_steps):
        return
    seed = int(getattr(env_setup, "problem_seed", config.seed))
    rl_eval.run(
        current=step,
        interval=int(config.eval_interval_steps),
        seed=seed,
        eval_seed_base=config.eval_seed_base,
        eval_noise_mode=getattr(config, "eval_noise_mode", None),
        state=train_state,
        evaluate_actor=lambda *, eval_seed: evaluate_actor(config, env_setup, modules, device=device, eval_seed=eval_seed),
        capture_actor_state=lambda: capture_actor_state(modules),
        evaluate_heldout=lambda *, best_actor_state, heldout_i_noise: evaluate_heldout(
            config,
            env_setup,
            modules,
            train_state,
            device=device,
            heldout_i_noise=heldout_i_noise,
        ),
    )
    now = float(time.time())
    rec = sac_metrics.record(
        step=int(step),
        eval_return=float(train_state.last_eval_return),
        heldout_return=train_state.last_heldout_return,
        best_return=float(train_state.best_return),
        loss_actor=float(latest_losses["loss_actor"]),
        loss_critic=float(latest_losses["loss_critic"]),
        loss_alpha=float(latest_losses["loss_alpha"]),
        total_updates=int(total_updates),
        started_at=float(start_time),
        now=now,
    )
    rl_logger.append_metrics(training_setup.metrics_path, rec)


def log_if_due(
    config: Any,
    train_state: Any,
    *,
    step: int,
    start_time: float,
    latest_losses: dict[str, float],
    total_updates: int,
) -> None:
    if not is_due(step, config.log_interval_steps):
        return
    now = float(time.time())
    line = sac_metrics.log(
        step=int(step),
        frames_per_batch=1,
        started_at=float(start_time),
        now=now,
        eval_return=float(train_state.last_eval_return),
        heldout_return=train_state.last_heldout_return,
        best_return=float(train_state.best_return),
        loss_actor=float(latest_losses["loss_actor"]),
        loss_critic=float(latest_losses["loss_critic"]),
        loss_alpha=float(latest_losses["loss_alpha"]),
    )
    rl_logger.log_eval_iteration(**line)


def checkpoint_if_due(
    config: Any,
    modules: Any,
    training_setup: Any,
    train_state: Any,
    *,
    step: int,
    build_checkpoint_payload: Any,
) -> None:
    save_if_due(
        training_setup.checkpoint_manager,
        build_checkpoint_payload(modules, training_setup, train_state, step=step),
        iteration=step,
        interval=config.checkpoint_interval_steps,
    )


def save_final_checkpoint_if_enabled(
    config: Any,
    modules: Any,
    training_setup: Any,
    train_state: Any,
    *,
    build_checkpoint_payload: Any,
) -> None:
    save_final_if_enabled(
        training_setup.checkpoint_manager,
        build_checkpoint_payload(modules, training_setup, train_state, step=int(config.total_timesteps)),
        iteration=int(config.total_timesteps),
        interval=config.checkpoint_interval_steps,
    )
