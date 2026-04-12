from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
from dm_control import suite

from problems.dm_control_spaces import flatten_obs, spec_to_space


def configure_headless_render_backend(render_mode: str | None) -> None:
    if render_mode != "rgb_array":
        return
    if not sys.platform.startswith("linux"):
        return
    if "MUJOCO_GL" in os.environ:
        return
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if has_display:
        return
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def parse_env_name(env_name: str) -> Tuple[str, str]:
    if not env_name.startswith("dm_control/"):
        raise ValueError(f"Expected dm_control env name, got: {env_name}")
    name = env_name.split("/", 1)[1]
    if name.endswith("-v0") or name.endswith("-v1"):
        name, _version = name.rsplit("-", 1)
    if "-" not in name:
        raise ValueError(f"Expected dm_control/<domain>-<task>-v0, got: {env_name}")
    domain, task = name.split("-", 1)
    return domain, task


DEFAULT_RENDER_WIDTH = 1280
DEFAULT_RENDER_HEIGHT = 720
PIXEL_HEIGHT = 84
PIXEL_WIDTH = 84
_PREFERRED_DM_CAMERA_NAMES = (
    "close",
    "near",
    "follow",
    "track",
    "side_close",
    "side",
    "front_close",
    "front",
)


class DMControlEnv:
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        domain: str,
        task: str,
        *,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        self._domain = domain
        self._task = task
        self._seed = seed
        self.render_mode = render_mode
        configure_headless_render_backend(render_mode)
        self._env = self._load_env(seed)
        self._render_width, self._render_height = self._resolve_render_size()
        self._camera_id = self._resolve_camera_id()
        self._obs_spec = self._env.observation_spec()
        self._action_spec = self._env.action_spec()
        self.observation_space = spec_to_space(self._obs_spec)
        self.action_space = spec_to_space(self._action_spec)

    def _resolve_render_size(self) -> tuple[int, int]:
        width = int(DEFAULT_RENDER_WIDTH)
        height = int(DEFAULT_RENDER_HEIGHT)
        try:
            model_global = self._env.physics.model.vis.global_
            offwidth = int(getattr(model_global, "offwidth", width))
            offheight = int(getattr(model_global, "offheight", height))
            if offwidth < width:
                try:
                    model_global.offwidth = int(width)
                except Exception:
                    pass
            if offheight < height:
                try:
                    model_global.offheight = int(height)
                except Exception:
                    pass
            offwidth = int(getattr(model_global, "offwidth", offwidth))
            offheight = int(getattr(model_global, "offheight", offheight))
            width = max(1, min(width, offwidth))
            height = max(1, min(height, offheight))
        except Exception:
            pass
        return width, height

    def _resolve_camera_id(self) -> int:
        model = getattr(getattr(self._env, "physics", None), "model", None)
        if model is None:
            return -1
        ncam = int(getattr(model, "ncam", 0))
        if ncam <= 0:
            return -1
        name2id = getattr(model, "name2id", None)
        if callable(name2id):
            for name in _PREFERRED_DM_CAMERA_NAMES:
                try:
                    cam_id = int(name2id(name, "camera"))
                except Exception:
                    continue
                if cam_id >= 0:
                    return cam_id
        cam_pos = None
        cam_mode = None
        cam_fovy = None
        try:
            cam_pos = np.asarray(getattr(model, "cam_pos"), dtype=np.float32)
            cam_mode = np.asarray(getattr(model, "cam_mode"), dtype=np.int32)
            cam_fovy = np.asarray(getattr(model, "cam_fovy"), dtype=np.float32)
        except Exception:
            return 0

        if cam_pos.ndim != 2 or cam_pos.shape[0] < ncam or cam_pos.shape[1] < 3:
            return 0
        if cam_mode.ndim != 1 or cam_mode.shape[0] < ncam:
            cam_mode = np.zeros((ncam,), dtype=np.int32)
        if cam_fovy.ndim != 1 or cam_fovy.shape[0] < ncam:
            cam_fovy = np.full((ncam,), 45.0, dtype=np.float32)

        dist = np.linalg.norm(cam_pos[:ncam, :3], axis=1)
        dist = np.where(np.isfinite(dist), dist, np.inf)
        fovy = np.where(np.isfinite(cam_fovy[:ncam]), cam_fovy[:ncam], 45.0)
        mode = cam_mode[:ncam]

        primary = np.where(mode != 0)[0]
        candidates = primary if primary.size > 0 else np.arange(ncam, dtype=np.int32)
        d = dist[candidates]
        target = float(np.median(d[np.isfinite(d)])) if np.any(np.isfinite(d)) else 0.0
        scores = np.abs(d - target) + 0.01 * fovy[candidates]
        best_local = int(np.argmin(scores))
        best_cam = int(candidates[best_local])
        return max(0, best_cam)

    def _load_env(self, seed: int | None):
        task_kwargs = {"random": seed} if seed is not None else None
        return suite.load(self._domain, self._task, task_kwargs=task_kwargs)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        _ = options
        if seed is not None and seed != self._seed:
            self._seed = seed
            self._env = self._load_env(seed)
            self._render_width, self._render_height = self._resolve_render_size()
        time_step = self._env.reset()
        obs = flatten_obs(time_step.observation)
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=self._action_spec.dtype)
        time_step = self._env.step(action)
        obs = flatten_obs(time_step.observation)
        reward = float(time_step.reward) if time_step.reward is not None else 0.0
        terminated = bool(time_step.last())
        truncated = False
        return obs, reward, terminated, truncated, {"discount": time_step.discount}

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        return self._env.physics.render(
            width=int(self._render_width),
            height=int(self._render_height),
            camera_id=int(self._camera_id),
        )

    def close(self):
        try:
            self._env.close()
        except Exception:
            return
