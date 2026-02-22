from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


def scale_action_to_space(action: np.ndarray | int, action_space: Any) -> np.ndarray | int:
    """Map policy output to action space. Discrete: pass through as int. Box: scale from [-1,1]."""
    if not hasattr(action_space, "low"):
        if hasattr(action_space, "n"):  # Discrete
            return int(action) if isinstance(action, (int, float, np.integer)) else int(np.asarray(action).item())
        return action
    action = np.asarray(action, dtype=np.float64)
    return action_space.low + (action_space.high - action_space.low) * (1 + action) / 2


def resolve_max_episode_steps(env_conf: Any) -> int:
    if getattr(env_conf, "gym_conf", None) is not None:
        return int(env_conf.gym_conf.max_steps)
    return int(getattr(env_conf, "max_steps", 99999))


def rollout_episode(
    env_conf: Any,
    policy: Any,
    *,
    seed: int,
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
            episode_trigger=lambda _episode: True,
            disable_logger=True,
        )

    state, _ = env.reset(seed=seed)
    total_return = 0.0
    lb, width = None, None
    if hasattr(env, "observation_space"):
        obs_space = env.observation_space
        low = getattr(obs_space, "low", None)
        high = getattr(obs_space, "high", None)
        if low is not None and high is not None and np.all(np.isfinite(low)) and np.all(np.isfinite(np.asarray(high) - np.asarray(low))):
            lb = np.asarray(low, dtype=np.float64)
            width = np.maximum(np.asarray(high, dtype=np.float64) - lb, 1e-8)
    for _ in range(resolve_max_episode_steps(env_conf)):
        state_np = np.asarray(state, dtype=np.float32)
        state_scaled = (state_np - lb) / width if (lb is not None and width is not None) else state_np
        action = policy(state_scaled)
        action = scale_action_to_space(action, env.action_space)
        state, reward, terminated, truncated, _ = env.step(action)
        total_return += float(reward)
        if terminated or truncated:
            break
    env.close()
    return float(total_return)


def select_video_episode_indices(
    episode_returns: list[float],
    *,
    selection: str,
    num_video_episodes: int,
    base_seed: int,
) -> list[int]:
    if num_video_episodes <= 0:
        return []
    count = min(int(num_video_episodes), len(episode_returns))
    if selection == "first":
        return list(range(count))
    if selection == "random":
        rng = np.random.default_rng(base_seed)
        return list(rng.choice(len(episode_returns), size=count, replace=False))
    ranked = sorted(range(len(episode_returns)), key=lambda i: episode_returns[i], reverse=True)
    return ranked[:count]


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
    if getattr(env_conf, "gym_conf", None) is None:
        return

    selection = str(episode_selection).lower()
    if selection not in ("best", "first", "random"):
        raise ValueError("episode_selection must be one of: best, first, random")

    video_dir.mkdir(parents=True, exist_ok=True)
    episode_returns = [
        rollout_episode(
            env_conf,
            policy,
            seed=seed_base + episode_idx,
            render_video=False,
            video_dir=None,
            video_prefix=video_prefix,
        )
        for episode_idx in range(num_episodes)
    ]
    selected_indices = select_video_episode_indices(
        episode_returns,
        selection=selection,
        num_video_episodes=num_video_episodes,
        base_seed=seed_base,
    )
    print(
        f"[video] dir={video_dir} episodes={num_episodes} videos={len(selected_indices)} select={selection}",
        flush=True,
    )
    for episode_idx in selected_indices:
        rollout_episode(
            env_conf,
            policy,
            seed=seed_base + episode_idx,
            render_video=True,
            video_dir=video_dir,
            video_prefix=f"{video_prefix}_ep{episode_idx:03d}",
        )


def policy_for_bo_rollout(env_conf: Any, policy: Any) -> Any:
    env_conf.ensure_spaces()
    g = getattr(env_conf, "gym_conf", None)
    recurrent = getattr(policy, "_recurrent", False)
    if g is None or not getattr(g, "transform_state", False):

        def _plain(s, **kwargs):
            return np.asarray(policy(s, **kwargs), dtype=np.float32)

        _plain._recurrent = recurrent
        return _plain

    lb = np.asarray(g.state_space.low, dtype=np.float32)
    width = np.asarray(g.state_space.high - g.state_space.low, dtype=np.float32)
    width = np.maximum(width, 1e-8)

    def wrapped(state: np.ndarray, **kwargs) -> np.ndarray:
        state_norm = (np.asarray(state, dtype=np.float32) - lb) / width
        return np.asarray(policy(state_norm, **kwargs), dtype=np.float32)

    wrapped._recurrent = recurrent
    return wrapped


def render_policy_videos_bo(
    env_conf: Any,
    policy: Any,
    *,
    video_dir: Path | str,
    video_prefix: str,
    num_episodes: int,
    num_video_episodes: int,
    episode_selection: str,
    seed_base: int,
) -> None:
    effective_policy = policy_for_bo_rollout(env_conf, policy)
    render_policy_videos(
        env_conf,
        effective_policy,
        video_dir=Path(video_dir),
        video_prefix=str(video_prefix),
        num_episodes=int(num_episodes),
        num_video_episodes=int(num_video_episodes),
        episode_selection=str(episode_selection),
        seed_base=int(seed_base),
    )


@dataclass(frozen=True)
class RLVideoContext:
    build_eval_env_conf: Callable[[int, int], Any]
    make_eval_policy: Callable[[Any, Any], Any]
    capture_actor_state: Callable[[Any], dict]
    with_actor_state: Callable[..., AbstractContextManager[Any]]


def render_policy_videos_rl(
    config: Any,
    env_setup: Any,
    modules: Any,
    training_setup: Any,
    train_state: Any,
    context: RLVideoContext,
    *,
    device: Any,
) -> None:
    if not config.video_enable:
        return

    eval_env_conf = context.build_eval_env_conf(int(env_setup.problem_seed), int(env_setup.noise_seed_0))
    if eval_env_conf.gym_conf is None:
        print(
            f"[rl/ppo/torchrl] video disabled for non-gym env: {config.env_tag}",
            flush=True,
        )
        return

    video_dir = training_setup.exp_dir / "videos"

    base_seed = int(
        config.video_seed_base if config.video_seed_base is not None else (config.eval_seed_base if config.eval_seed_base is not None else config.seed)
    )
    actor_state = train_state.best_actor_state if train_state.best_actor_state is not None else context.capture_actor_state(modules)
    with context.with_actor_state(modules, actor_state, device=device):
        eval_policy = context.make_eval_policy(modules, device)
        render_policy_videos(
            eval_env_conf,
            eval_policy,
            video_dir=video_dir,
            video_prefix=str(config.video_prefix),
            num_episodes=int(config.video_num_episodes),
            num_video_episodes=int(config.video_num_video_episodes),
            episode_selection=str(config.video_episode_selection or "best"),
            seed_base=base_seed,
        )
