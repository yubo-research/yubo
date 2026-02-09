from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _replace_non_finite(arr: np.ndarray, *, fill: float) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32).copy()
    values[~np.isfinite(values)] = fill
    return values


def _max_steps(env_conf: Any) -> int:
    if getattr(env_conf, "gym_conf", None) is not None:
        return int(env_conf.gym_conf.max_steps)
    return 99999


def _transform_action(action_p: np.ndarray, action_space: Any) -> np.ndarray:
    if not hasattr(action_space, "low"):
        return action_p
    return action_space.low + (action_space.high - action_space.low) * (1 + action_p) / 2


def _sanitize_array(values: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0):
    return np.nan_to_num(np.asarray(values, dtype=np.float64), nan=nan, posinf=posinf, neginf=neginf)


def _rollout_episode(
    env_conf: Any,
    policy: Any,
    *,
    seed: int,
    transform_state: bool,
    lb: np.ndarray | None,
    width: np.ndarray | None,
    render_video: bool,
    video_dir: Path | None,
    video_prefix: str,
) -> float:
    render_mode = "rgb_array" if render_video else None
    env = env_conf.make(render_mode=render_mode)
    if render_video:
        import gymnasium as gym

        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            name_prefix=video_prefix,
            episode_trigger=lambda _ep: True,
            disable_logger=True,
        )
    if hasattr(policy, "reset_state"):
        policy.reset_state()
    state, _ = env.reset(seed=seed)
    total_return = 0.0
    for _ in range(_max_steps(env_conf)):
        state_f = _sanitize_array(state)
        state_p = state_f
        if transform_state:
            assert lb is not None and width is not None
            state_p = (state_f - lb) / width
            state_p = _sanitize_array(state_p)
        policy_input = state_p if transform_state else state_f
        action_p = _sanitize_array(policy(policy_input), posinf=1.0, neginf=-1.0)
        action_p = np.clip(action_p, -1.0, 1.0)
        action = _transform_action(action_p, env.action_space)
        step_out = env.step(action)
        if len(step_out) == 5:
            state, reward, terminated, truncated, _ = step_out
            done = bool(terminated or truncated)
        else:
            state, reward, done = step_out[:3]
        total_return += float(reward)
        if done:
            break
    env.close()
    return float(total_return)


def _select_video_indices(
    returns: list[float],
    *,
    selection: str,
    num_video_episodes: int,
    base_seed: int,
) -> list[int]:
    if num_video_episodes <= 0:
        return []
    num_video_episodes = min(num_video_episodes, len(returns))
    if selection == "first":
        return list(range(num_video_episodes))
    if selection == "random":
        rng = np.random.default_rng(base_seed)
        return list(rng.choice(len(returns), size=num_video_episodes, replace=False))
    ranked = sorted(range(len(returns)), key=lambda i: returns[i], reverse=True)
    return ranked[:num_video_episodes]


def render_policy_videos(
    env_conf: Any,
    policy: Any,
    *,
    video_dir: Path,
    video_prefix: str,
    num_episodes: int,
    num_video_episodes: int,
    episode_selection: str,
    seed_base: int,
) -> None:
    env_conf.ensure_spaces()
    obs_space = env_conf.gym_conf.state_space
    transform_state = bool(env_conf.gym_conf.transform_state)
    lb = None
    width = None
    if transform_state:
        lb = _replace_non_finite(np.asarray(obs_space.low, dtype=np.float32), fill=0.0)
        ub = _replace_non_finite(np.asarray(obs_space.high, dtype=np.float32), fill=1.0)
        width = ub - lb
        width[~np.isfinite(width)] = 1.0
        width[width == 0.0] = 1.0

    selection = str(episode_selection or "best").lower()
    if selection not in ("best", "first", "random"):
        raise ValueError("episode_selection must be one of: best, first, random")

    returns = []
    for i in range(int(num_episodes)):
        seed = int(seed_base + i)
        returns.append(
            _rollout_episode(
                env_conf,
                policy,
                seed=seed,
                transform_state=transform_state,
                lb=lb,
                width=width,
                render_video=False,
                video_dir=None,
                video_prefix=video_prefix,
            )
        )

    indices = _select_video_indices(
        returns,
        selection=selection,
        num_video_episodes=int(num_video_episodes),
        base_seed=int(seed_base),
    )
    if not indices:
        return

    for idx in indices:
        seed = int(seed_base + idx)
        prefix = f"{video_prefix}_ep{idx:03d}"
        _rollout_episode(
            env_conf,
            policy,
            seed=seed,
            transform_state=transform_state,
            lb=lb,
            width=width,
            render_video=True,
            video_dir=video_dir,
            video_prefix=prefix,
        )
