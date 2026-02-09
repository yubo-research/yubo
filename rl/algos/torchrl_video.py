from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


def scale_action_to_space(action: np.ndarray, action_space: Any) -> np.ndarray:
    if not hasattr(action_space, "low"):
        return action
    return action_space.low + (action_space.high - action_space.low) * (1 + action) / 2


def resolve_max_episode_steps(env_conf: Any) -> int:
    if getattr(env_conf, "gym_conf", None) is not None:
        return int(env_conf.gym_conf.max_steps)
    return 99999


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
    for _ in range(resolve_max_episode_steps(env_conf)):
        action = policy(state)
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


def render_best_policy_videos(
    config: Any,
    env_setup: Any,
    modules: Any,
    training_setup: Any,
    train_state: Any,
    *,
    device: torch.device,
    build_eval_env_conf: Callable[[int, int], Any],
    eval_policy_factory: Callable[[Any, torch.device], Any],
    capture_actor_state: Callable[[Any], dict],
    temporary_actor_state: Callable[..., Any],
) -> None:
    if not config.video_enable:
        return

    eval_env_conf = build_eval_env_conf(int(env_setup.problem_seed), int(env_setup.noise_seed_0))
    if eval_env_conf.gym_conf is None:
        print(f"[rl/ppo/torchrl] video disabled for non-gym env: {config.env_tag}", flush=True)
        return

    selection = str(config.video_episode_selection or "best").lower()
    if selection not in ("best", "first", "random"):
        raise ValueError("video_episode_selection must be one of: best, first, random")

    base_seed = int(
        config.video_seed_base if config.video_seed_base is not None else (config.eval_seed_base if config.eval_seed_base is not None else config.seed)
    )
    num_episodes = int(config.video_num_episodes)
    num_video_episodes = int(config.video_num_video_episodes)

    video_dir = Path(config.video_dir) if config.video_dir else (training_setup.exp_dir / "videos")
    if not video_dir.is_absolute():
        video_dir = training_setup.exp_dir / video_dir
    video_dir.mkdir(parents=True, exist_ok=True)

    actor_state = train_state.best_actor_state if train_state.best_actor_state is not None else capture_actor_state(modules)
    with temporary_actor_state(modules, actor_state, device=device):
        eval_policy = eval_policy_factory(modules, device)
        episode_returns = [
            rollout_episode(
                eval_env_conf,
                eval_policy,
                seed=base_seed + episode_idx,
                render_video=False,
                video_dir=None,
                video_prefix=config.video_prefix,
            )
            for episode_idx in range(num_episodes)
        ]

        selected_indices = select_video_episode_indices(
            episode_returns,
            selection=selection,
            num_video_episodes=num_video_episodes,
            base_seed=base_seed,
        )
        print(
            f"[rl/ppo/torchrl] videos dir={video_dir} episodes={num_episodes} videos={len(selected_indices)} select={selection}",
            flush=True,
        )
        for episode_idx in selected_indices:
            rollout_episode(
                eval_env_conf,
                eval_policy,
                seed=base_seed + episode_idx,
                render_video=True,
                video_dir=video_dir,
                video_prefix=f"{config.video_prefix}_ep{episode_idx:03d}",
            )
