"""DM Control environments via Shimmy (Gymnasium API). Replaces direct dm_control usage."""

from __future__ import annotations

import importlib
import os
import sys
import warnings
from typing import Any

import gymnasium as gym
import numpy as np


def _ensure_platform_ready() -> None:
    """Set MUJOCO_GL so dm_control can run on this platform."""
    if "MUJOCO_GL" in os.environ:
        return
    if sys.platform == "darwin":
        os.environ.setdefault("MUJOCO_GL", "glfw")
    elif sys.platform.startswith("linux"):
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not has_display:
            os.environ.setdefault("MUJOCO_GL", "egl")
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


_ensure_platform_ready()

PIXEL_HEIGHT = 84
PIXEL_WIDTH = 84


def _parse_env_name(env_name: str) -> tuple[str, str]:
    if not env_name.startswith("dm_control/"):
        raise ValueError(f"Expected dm_control env name, got: {env_name}")
    name = env_name.split("/", 1)[1]
    if name.endswith("-v0") or name.endswith("-v1"):
        name = name.rsplit("-", 1)[0]
    if "-" not in name:
        raise ValueError(f"Expected dm_control/<domain>-<task>-v0, got: {env_name}")
    domain, task = name.split("-", 1)
    return domain, task


def _flatten_obs(obs: dict | np.ndarray) -> np.ndarray:
    """Flatten dict obs to 1D vector (for state-only, BO compatibility)."""
    if isinstance(obs, np.ndarray):
        return np.ravel(obs.astype(np.float32))
    parts = [np.ravel(np.asarray(obs[k], dtype=np.float32)) for k in sorted(obs)]
    return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float32)


def _resize_pixels(pixels: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    from scipy.ndimage import zoom

    h, w = pixels.shape[:2]
    factors = (target_h / h, target_w / w, 1.0)
    out = zoom(pixels, factors, order=0)
    return np.asarray(out, dtype=np.uint8)


def _is_gl_init_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "gladloadgl",
            "opengl platform library",
            "valid opengl context",
            "mjr_makecontext",
            "glfw library is not initialized",
        )
    )


class _PixelObsWrapper(gym.Wrapper):
    """Replace state obs with rendered pixels. For from_pixels mode."""

    def __init__(self, env: gym.Env, *, pixels_only: bool = True, size: int = 84):
        super().__init__(env)
        self._pixels_only = pixels_only
        self._size = size
        self._render_disabled = False
        pixel_space = gym.spaces.Box(low=0, high=255, shape=(size, size, 3), dtype=np.uint8)
        if pixels_only:
            self.observation_space = gym.spaces.Dict({"pixels": pixel_space})
            return
        state_space = env.observation_space
        if isinstance(state_space, gym.spaces.Dict):
            state_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(sum(np.prod(s.shape) for s in state_space.spaces.values()),),
                dtype=np.float32,
            )
        self.observation_space = gym.spaces.Dict({"pixels": pixel_space, "state": state_space})

    def _get_pixels(self) -> np.ndarray:
        if self._render_disabled:
            return np.zeros((self._size, self._size, 3), dtype=np.uint8)
        try:
            img = self.env.render()
        except Exception as exc:
            if not _is_gl_init_error(exc):
                raise
            self._render_disabled = True
            warnings.warn(
                "dm_control render backend unavailable; returning zero-valued pixel observations.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.zeros((self._size, self._size, 3), dtype=np.uint8)
        if img is None:
            return np.zeros((self._size, self._size, 3), dtype=np.uint8)
        return _resize_pixels(img, self._size, self._size)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        pixels = self._get_pixels()
        if self._pixels_only:
            return {"pixels": pixels}, info
        return {"pixels": pixels, "state": _flatten_obs(obs)}, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pixels = self._get_pixels()
        if self._pixels_only:
            return {"pixels": pixels}, reward, terminated, truncated, info
        return (
            {"pixels": pixels, "state": _flatten_obs(obs)},
            reward,
            terminated,
            truncated,
            info,
        )


class _FlattenDictObsWrapper(gym.ObservationWrapper):
    """Flatten dict obs to 1D for BO compatibility (state-only)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if isinstance(env.observation_space, gym.spaces.Dict):
            low = np.concatenate([np.ravel(env.observation_space.spaces[k].low) for k in sorted(env.observation_space.spaces)])
            high = np.concatenate([np.ravel(env.observation_space.spaces[k].high) for k in sorted(env.observation_space.spaces)])
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        return _flatten_obs(obs)


def make(
    env_name: str,
    *,
    render_mode: str | None = None,
    from_pixels: bool = False,
    pixels_only: bool = True,
    **kwargs: Any,
) -> gym.Env:
    """Create dm_control env via Shimmy. Same interface as dm_control_env.make."""
    _ensure_platform_ready()
    importlib.import_module("shimmy")
    domain, task = _parse_env_name(env_name)
    gym_id = f"dm_control/{domain}-{task}-v0"
    render = "rgb_array" if from_pixels else render_mode
    base = gym.make(gym_id, render_mode=render, **kwargs)
    if from_pixels:
        return _PixelObsWrapper(base, pixels_only=pixels_only, size=PIXEL_HEIGHT)
    return _FlattenDictObsWrapper(base)
