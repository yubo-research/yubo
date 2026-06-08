from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from problems.isaaclab_env_adapters import (
    get_isaaclab_session,
    is_isaaclab_env_tag,
    isaaclab_video_launcher_kwargs,
)
from video.isaaclab_viewport import capture_isaaclab_viewport_frame
from video.rollout import _open_frame_video_writer, _rollout_episode_body


def is_isaaclab_env_conf(env_conf: Any) -> bool:
    return is_isaaclab_env_tag(str(getattr(env_conf, "env_name", "")))


def ensure_isaaclab_video_launcher(env_conf: Any) -> None:
    kwargs = dict(getattr(env_conf, "kwargs", {}) or {})
    launcher_kwargs = kwargs.get("launcher_kwargs")
    kit_args = str(launcher_kwargs.get("kit_args", "")) if isinstance(launcher_kwargs, dict) else ""
    if "omni.kit.viewport.utility" not in kit_args:
        kwargs["launcher_kwargs"] = isaaclab_video_launcher_kwargs()
    spec = getattr(env_conf, "spec", None)
    if spec is not None and hasattr(spec, "kwargs"):
        spec.kwargs = kwargs
        return
    env_conf.kwargs = kwargs


class _ViewportRenderEnv:
    def __init__(self, env: Any, raw_env: Any, tmp_dir: Path) -> None:
        self._env = env
        self._raw_env = raw_env
        self._tmp_dir = tmp_dir
        self._frame_index = 0
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def step(self, action):
        return self._env.step(action)

    def render(self):
        frame_path = self._tmp_dir / f"frame_{self._frame_index:06d}.png"
        self._frame_index += 1
        return capture_isaaclab_viewport_frame(get_isaaclab_session().app, self._raw_env, frame_path)


def render_isaaclab_video_episode(
    env_conf: Any,
    policy: Any,
    *,
    seed: int,
    video_dir: Path,
    video_prefix: str,
) -> float:
    ensure_isaaclab_video_launcher(env_conf)
    env = env_conf.make(render_mode=None, launcher_kwargs=isaaclab_video_launcher_kwargs())
    raw_env = getattr(env, "_env", env)
    video_path = Path(video_dir) / f"{video_prefix}-episode-0.mp4"
    writer = _open_frame_video_writer(video_path)
    try:
        with tempfile.TemporaryDirectory(prefix="isaaclab-video-") as tmp:
            wrapped = _ViewportRenderEnv(env, raw_env, Path(tmp))
            return _rollout_episode_body(env_conf, wrapped, policy, seed=int(seed), frame_writer=writer)
    finally:
        writer.close()
        env.close()
