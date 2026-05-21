from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from video.spaces import resolve_max_episode_steps, scale_action_to_space


def _use_gym_record_video(env: Any) -> bool:
    try:
        import gymnasium as gym
    except Exception:
        return False
    return isinstance(env, gym.Env)


def _video_output_path(video_dir: Path, video_prefix: str) -> Path:
    return Path(video_dir) / f"{video_prefix}-episode-0.mp4"


def _frame_to_uint8(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3:
        raise ValueError(f"Expected RGB frame with 3 dims, got shape={arr.shape}.")
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        max_val = float(np.max(arr)) if arr.size > 0 else 1.0
        if max_val <= 1.0:
            arr = arr * 255.0
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def _open_frame_video_writer(video_path: Path, *, fps: int = 30):
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("imageio is required for non-gym video capture.") from exc
    video_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return imageio.get_writer(str(video_path), fps=int(fps))
    except Exception as exc:
        raise RuntimeError(f"Failed to create video writer for '{video_path}'.") from exc


def _rollout_episode_body(env_conf: Any, env: Any, policy: Any, *, seed: int, frame_writer: Any) -> float:
    state, _ = env.reset(seed=seed)
    if frame_writer is not None:
        frame_writer.append_data(_frame_to_uint8(env.render()))
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
        if frame_writer is not None:
            frame_writer.append_data(_frame_to_uint8(env.render()))
        total_return += float(reward)
        if terminated or truncated:
            break
    return float(total_return)


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
    frame_writer = None
    if render_video:
        if _use_gym_record_video(env):
            import gymnasium as gym

            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(video_dir),
                name_prefix=video_prefix,
                episode_trigger=lambda _episode: True,
                disable_logger=True,
            )
        else:
            if video_dir is None:
                raise ValueError("video_dir must be provided when render_video=True.")
            frame_writer = _open_frame_video_writer(_video_output_path(video_dir, video_prefix))

    try:
        return _rollout_episode_body(env_conf, env, policy, seed=seed, frame_writer=frame_writer)
    finally:
        if frame_writer is not None:
            frame_writer.close()
        env.close()
