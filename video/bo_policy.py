from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from video.batch import render_policy_videos


def policy_for_bo_rollout(env_conf: Any, policy: Any) -> Any:
    env_conf.ensure_spaces()
    g = getattr(env_conf, "gym_conf", None)
    recurrent = getattr(policy, "_recurrent", False)
    if g is None or not getattr(g, "transform_state", False):

        def _plain(s, **kwargs):
            return np.asarray(policy(s, **kwargs), dtype=np.float32)

        _plain._recurrent = recurrent
        return _plain

    low = np.asarray(g.state_space.low, dtype=np.float32)
    high = np.asarray(g.state_space.high, dtype=np.float32)
    width_raw = np.asarray(high - low, dtype=np.float32)
    bounded = np.isfinite(low) & np.isfinite(high) & np.isfinite(width_raw) & (width_raw > 0)
    lb = np.where(bounded, low, 0.0).astype(np.float32)
    width = np.where(bounded, np.maximum(width_raw, 1e-8), 1.0).astype(np.float32)

    def wrapped(state: np.ndarray, **kwargs) -> np.ndarray:
        state_arr = np.asarray(state, dtype=np.float32)
        state_norm = np.asarray(state_arr, dtype=np.float32).copy()
        if state_norm.shape == bounded.shape:
            state_norm[bounded] = (state_arr[bounded] - lb[bounded]) / width[bounded]
        else:
            # Fallback for unexpected shape mismatch: preserve previous behavior,
            # then sanitize to avoid NaN/Inf driving unstable control.
            state_norm = (state_arr - lb) / width
        state_norm = np.nan_to_num(state_norm, nan=0.0, posinf=1e6, neginf=-1e6)
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
