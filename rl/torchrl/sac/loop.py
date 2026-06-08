from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from rl import logger
from rl.core.progress import due_mark, is_due
from rl.iter_record import IterInputs


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
    if step < int(config.collector.init_random_frames):
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
    if step < int(config.collector.init_random_frames) or step % int(config.optim.update_every) != 0:
        return (latest_losses, total_updates)
    for _ in range(int(config.optim.optim_steps_per_batch)):
        latest_losses = update_step(
            training_setup,
            device=device,
            batch_size=int(config.replay_buffer.batch_size),
        )
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
    evaluate_for_best: Any,
    build_env_runtime: Any | None = None,
    get_env_conf: Any | None = None,
    heldout_i_noise: int = 99999,
) -> float | None:
    sac_eval = __import__("rl.core.sac_eval", fromlist=["evaluate_heldout_with_best_actor"])
    best_eval_policy = eval_policy_factory(modules, env_setup, device)
    if build_env_runtime is not None:
        env_conf = build_env_runtime(
            config.env_tag,
            problem_seed=env_setup.problem_seed,
            noise_seed_0=env_setup.noise_seed_0,
            from_pixels=bool(getattr(getattr(env_setup, "env_conf", None), "from_pixels", False)),
            pixels_only=bool(getattr(getattr(env_setup, "env_conf", None), "pixels_only", True)),
        )
    elif get_env_conf is not None:
        env_conf = get_env_conf(
            config.env_tag,
            problem_seed=env_setup.problem_seed,
            noise_seed_0=env_setup.noise_seed_0,
        )
    else:
        raise TypeError("evaluate_heldout_if_enabled requires build_env_runtime or get_env_conf")
    return sac_eval.evaluate_heldout_with_best_actor(
        best_actor_state=train_state.best_actor_state,
        num_denoise_passive=config.eval.num_denoise_passive,
        heldout_i_noise=int(heldout_i_noise),
        with_actor_state=lambda snapshot: temporary_actor_state(
            modules,
            snapshot,
            capture_actor_state=capture_actor_state,
            restore_actor_state=restore_actor_state,
        ),
        evaluate_for_best=evaluate_for_best,
        eval_env_conf=env_conf,
        eval_policy=best_eval_policy,
    )


def eval_and_log_if_due(
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
    iter_dt: float,
    n_frames: int,
    log_on_interval: bool = True,
) -> None:
    from rl.eval_noise import build_eval_plan
    from rl.torchrl_metrics import build_sac_iter_record

    did_eval = False
    eval_dt = None
    eval_mark = _advance_due_mark(train_state, "_last_eval_mark", step, config.eval.interval_steps)
    if eval_mark is not None:
        eval_interval = int(config.eval.interval_steps)
        run_problem_seed = int(getattr(env_setup, "problem_seed", config.seed))
        plan = build_eval_plan(
            current=int(eval_mark * eval_interval),
            interval=eval_interval,
            seed=run_problem_seed,
            eval_seed_base=config.eval.seed_base,
            eval_noise_mode=config.eval.noise_mode,
        )
        eval_start = time.time()
        train_state.last_eval_return = evaluate_actor(
            config,
            env_setup,
            modules,
            device=device,
            eval_seed=plan.eval_seed,
        )
        sac_eval = __import__("rl.core.sac_eval", fromlist=["update_best_actor_if_improved"])
        train_state.best_return, train_state.best_actor_state, _ = sac_eval.update_best_actor_if_improved(
            eval_return=float(train_state.last_eval_return),
            best_return=float(train_state.best_return),
            best_actor_state=train_state.best_actor_state,
            capture_actor_state=lambda: capture_actor_state(modules),
        )
        train_state.last_heldout_return = evaluate_heldout(
            config,
            env_setup,
            modules,
            train_state,
            device=device,
            heldout_i_noise=plan.heldout_i_noise,
        )
        eval_dt = time.time() - eval_start
        did_eval = True

    log_due = False
    if log_on_interval:
        log_mark = _advance_due_mark(train_state, "_last_log_mark", step, config.log_interval_steps)
        log_due = log_mark is not None
    if not (did_eval or log_due):
        return

    elapsed = time.time() - start_time
    record = build_sac_iter_record(
        _sac_iter_inputs(
            step,
            max(1, int(n_frames)),
            elapsed,
            iter_dt,
            latest_losses,
            train_state,
            total_updates,
            eval_dt=eval_dt,
        )
    )
    logger.log_rl_iter(record, metrics_path=training_setup.metrics_path)


def _sac_iter_inputs(
    step: int,
    frames_per_iter: int,
    elapsed: float,
    iter_dt: float,
    latest_losses: dict[str, float],
    train_state: Any,
    total_updates: int,
    *,
    eval_dt: float | None = None,
) -> IterInputs:
    frames = int(frames_per_iter)
    step_i = int(step)
    return IterInputs(
        iteration=step_i // max(1, frames),
        step=step_i,
        frames_per_iter=frames,
        elapsed=float(elapsed),
        iter_dt=float(iter_dt),
        metrics={
            "actor": float(latest_losses["loss_actor"]),
            "critic": float(latest_losses["loss_critic"]),
            "alpha": float(latest_losses["loss_alpha"]),
            "ret_best": float(train_state.best_return),
            "ret_eval": float(train_state.last_eval_return),
            "ret_heldout": getattr(train_state, "last_heldout_return", None),
            "eval_dt": eval_dt,
            "total_updates": int(total_updates),
        },
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
    iter_dt: float = 1e-9,
    n_frames: int = 1,
) -> None:
    eval_and_log_if_due(
        config,
        env_setup,
        modules,
        training_setup,
        train_state,
        step=step,
        device=device,
        start_time=start_time,
        latest_losses=latest_losses,
        total_updates=total_updates,
        evaluate_actor=evaluate_actor,
        capture_actor_state=capture_actor_state,
        evaluate_heldout=evaluate_heldout,
        iter_dt=iter_dt,
        n_frames=n_frames,
        log_on_interval=False,
    )


def log_if_due(
    config: Any,
    train_state: Any,
    *,
    step: int,
    start_time: float,
    latest_losses: dict[str, float],
    total_updates: int,
    iter_dt: float = 1e-9,
    n_frames: int = 1,
    training_setup: Any | None = None,
) -> None:
    from rl.torchrl_metrics import build_sac_iter_record

    if training_setup is None:
        return
    log_mark = _advance_due_mark(train_state, "_last_log_mark", step, config.log_interval_steps)
    if log_mark is None:
        return
    elapsed = time.time() - start_time
    record = build_sac_iter_record(
        _sac_iter_inputs(
            step,
            max(1, int(n_frames)),
            elapsed,
            iter_dt,
            latest_losses,
            train_state,
            total_updates,
        )
    )
    logger.log_rl_iter(record, metrics_path=training_setup.metrics_path)


def checkpoint_if_due(
    config: Any,
    modules: Any,
    training_setup: Any,
    train_state: Any,
    *,
    step: int,
    build_checkpoint_payload: Any,
) -> None:
    if _advance_due_mark(train_state, "_last_checkpoint_mark", step, config.checkpoint.interval_steps) is None:
        return
    payload = build_checkpoint_payload(modules, training_setup, train_state, step=step)
    training_setup.checkpoint_manager.save_both(payload, iteration=step)


def _advance_due_mark(train_state: Any, attr: str, step: int, interval: int | None) -> int | None:
    mark = due_mark(int(step), interval, int(getattr(train_state, attr, 0)))
    if mark is None:
        return None
    setattr(train_state, attr, int(mark))
    return int(mark)


def save_final_checkpoint_if_enabled(
    config: Any,
    modules: Any,
    training_setup: Any,
    train_state: Any,
    *,
    build_checkpoint_payload: Any,
) -> None:
    if not is_due(int(config.collector.total_frames), config.checkpoint.interval_steps):
        return
    payload = build_checkpoint_payload(modules, training_setup, train_state, step=int(config.collector.total_frames))
    training_setup.checkpoint_manager.save_both(payload, iteration=int(config.collector.total_frames))
