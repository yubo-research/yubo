from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable

from rl.core.rl_video_settings import get_video_settings
from video.batch import render_policy_videos


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
    video_settings = get_video_settings(config)
    if not video_settings.enable:
        return

    eval_env_conf = context.build_eval_env_conf(int(env_setup.problem_seed), int(env_setup.noise_seed_0))
    if eval_env_conf.gym_conf is None:
        print(
            f"[rl/ppo/torchrl] video disabled for non-gym env: {config.env_tag}",
            flush=True,
        )
        return

    video_dir = training_setup.exp_dir / "videos"

    seed_base = video_settings.seed_base if video_settings.seed_base is not None else getattr(config, "eval_seed_base", None)
    base_seed = int(seed_base if seed_base is not None else config.seed)
    actor_state = train_state.best_actor_state if train_state.best_actor_state is not None else context.capture_actor_state(modules)
    with context.with_actor_state(modules, actor_state, device=device):
        eval_policy = context.make_eval_policy(modules, device)
        render_policy_videos(
            eval_env_conf,
            eval_policy,
            video_dir=video_dir,
            video_prefix=str(video_settings.prefix),
            num_episodes=int(video_settings.num_episodes),
            num_video_episodes=int(video_settings.num_video_episodes),
            episode_selection=str(video_settings.episode_selection or "best"),
            seed_base=base_seed,
        )
