from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import torch
from tensordict import TensorDict

from rl.algos.checkpointing import append_jsonl


def is_due(step: int, interval: Optional[int]) -> bool:
    return interval is not None and int(interval) > 0 and step % int(interval) == 0


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
    if step < int(config.learning_starts):
        action_env = train_env.action_space.sample()
        action_norm = unscale_action_from_env(action_env, env_setup.action_low, env_setup.action_high)
        return action_env, action_norm

    with torch.no_grad():
        obs_td = TensorDict(
            {"observation": torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)},
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
        return reset_observation
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
) -> float | None:
    if config.num_denoise_passive_eval is None or train_state.best_actor_state is None:
        return None
    current_actor_state = capture_actor_state(modules)
    restore_actor_state(modules, train_state.best_actor_state)
    best_eval_policy = eval_policy_factory(modules, env_setup, device)
    heldout_return = float(
        evaluate_for_best(
            get_env_conf(config.env_tag, problem_seed=env_setup.problem_seed, noise_seed_0=env_setup.noise_seed_0),
            best_eval_policy,
            config.num_denoise_passive_eval,
        )
    )
    restore_actor_state(modules, current_actor_state)
    return heldout_return


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

    eval_seed = int(config.eval_seed_base if config.eval_seed_base is not None else config.seed)
    train_state.last_eval_return = evaluate_actor(config, env_setup, modules, device=device, eval_seed=eval_seed)
    if train_state.last_eval_return > train_state.best_return:
        train_state.best_return = float(train_state.last_eval_return)
        train_state.best_actor_state = capture_actor_state(modules)

    train_state.last_heldout_return = evaluate_heldout(config, env_setup, modules, train_state, device=device)
    elapsed = time.time() - start_time
    steps_per_second = float(step / elapsed) if elapsed > 0 else float("nan")
    append_jsonl(
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
    steps_per_second = float(step / elapsed) if elapsed > 0 else float("nan")
    eval_text = f"{train_state.last_eval_return:.2f}" if np.isfinite(train_state.last_eval_return) else "nan"
    heldout_text = "nan" if train_state.last_heldout_return is None else f"{train_state.last_heldout_return:.2f}"
    print(
        "[rl/sac/torchrl]"
        f" step={step:08d}/{config.total_timesteps}"
        f" eval={eval_text}"
        f" heldout={heldout_text}"
        f" best={train_state.best_return:.2f}"
        f" actor_loss={latest_losses['loss_actor']:.4f}"
        f" critic_loss={latest_losses['loss_critic']:.4f}"
        f" alpha_loss={latest_losses['loss_alpha']:.4f}"
        f" updates={total_updates}"
        f" sps={steps_per_second:.0f}",
        flush=True,
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
