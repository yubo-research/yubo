from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np
import torch
from tensordict import TensorDict

from rl import logger as rl_logger


def is_due(step: int, interval: Optional[int]) -> bool:
    return interval is not None and int(interval) > 0 and step % int(interval) == 0


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
        return action_env, action_norm

    with torch.no_grad():
        obs_td = TensorDict(
            {"observation": torch.as_tensor(obs_np, device=device).unsqueeze(0)},
            [1],
        )
        action_norm = modules.actor(obs_td)["action"][0].detach().cpu().numpy()
    action_env = scale_action_to_env(action_norm, env_setup.action_low, env_setup.action_high)
    return action_env, action_norm


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
        return latest_losses, total_updates
    for _ in range(int(config.updates_per_step)):
        latest_losses = update_step(training_setup, device=device, batch_size=int(config.batch_size))
        total_updates += 1
    return latest_losses, total_updates


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


def evaluate_heldout_if_enabled(
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
    evaluate_for_best: Any,
    heldout_i_noise: int = 99999,
) -> float | None:
    if config.num_denoise_passive_eval is None or train_state.best_actor_state is None:
        return None
    with temporary_actor_state(
        modules,
        train_state.best_actor_state,
        capture_actor_state=capture_actor_state,
        restore_actor_state=restore_actor_state,
    ):
        best_eval_policy = eval_policy_factory(modules, env_setup, device)
        env_conf = get_env_conf(
            config.env_tag,
            problem_seed=env_setup.problem_seed,
            noise_seed_0=env_setup.noise_seed_0,
            from_pixels=bool(getattr(getattr(env_setup, "env_conf", None), "from_pixels", False)),
            pixels_only=bool(getattr(getattr(env_setup, "env_conf", None), "pixels_only", True)),
        )
        return float(
            evaluate_for_best(
                env_conf,
                best_eval_policy,
                config.num_denoise_passive_eval,
                i_noise=heldout_i_noise,
            )
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

    from rl.eval_noise import build_eval_plan

    plan = build_eval_plan(
        current=step,
        interval=int(config.eval_interval_steps),
        seed=int(config.seed),
        eval_seed_base=config.eval_seed_base,
        eval_noise_mode=getattr(config, "eval_noise_mode", None),
    )
    train_state.last_eval_return = evaluate_actor(config, env_setup, modules, device=device, eval_seed=plan.eval_seed)
    if train_state.last_eval_return > train_state.best_return:
        train_state.best_return = float(train_state.last_eval_return)
        train_state.best_actor_state = capture_actor_state(modules)

    train_state.last_heldout_return = evaluate_heldout(
        config,
        env_setup,
        modules,
        train_state,
        device=device,
        heldout_i_noise=plan.heldout_i_noise,
    )
    elapsed = time.time() - start_time
    steps_per_second = float(step / elapsed) if elapsed > 0 else float("nan")
    rl_logger.append_metrics(
        training_setup.metrics_path,
        {
            "step": int(step),
            "eval_return": float(train_state.last_eval_return),
            "heldout_return": train_state.last_heldout_return,
            "best_return": float(train_state.best_return),
            "loss_actor": latest_losses["loss_actor"],
            "loss_critic": latest_losses["loss_critic"],
            "loss_alpha": latest_losses["loss_alpha"],
            "total_updates": int(total_updates),
            "time_seconds": elapsed,
            "steps_per_second": steps_per_second,
        },
    )


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

    elapsed = time.time() - start_time
    eval_return = train_state.last_eval_return if np.isfinite(train_state.last_eval_return) else None
    heldout_return = train_state.last_heldout_return
    best_return = train_state.best_return if np.isfinite(train_state.best_return) else 0.0
    rl_logger.log_eval_iteration(
        0,
        0,
        1,
        eval_return=eval_return,
        heldout_return=heldout_return,
        best_return=best_return,
        algo_metrics={
            "actor": latest_losses["loss_actor"],
            "critic": latest_losses["loss_critic"],
            "alpha": latest_losses["loss_alpha"],
        },
        algo_name="sac",
        elapsed=elapsed,
        step_override=step,
    )


def checkpoint_if_due(
    config: Any,
    modules: Any,
    training_setup: Any,
    train_state: Any,
    *,
    step: int,
    build_checkpoint_payload: Any,
) -> None:
    if not is_due(step, config.checkpoint_interval_steps):
        return
    payload = build_checkpoint_payload(modules, training_setup, train_state, step=step)
    training_setup.checkpoint_manager.save_both(payload, iteration=step)


def save_final_checkpoint_if_enabled(
    config: Any,
    modules: Any,
    training_setup: Any,
    train_state: Any,
    *,
    build_checkpoint_payload: Any,
) -> None:
    if not is_due(int(config.total_timesteps), config.checkpoint_interval_steps):
        return
    payload = build_checkpoint_payload(modules, training_setup, train_state, step=int(config.total_timesteps))
    training_setup.checkpoint_manager.save_both(payload, iteration=int(config.total_timesteps))
