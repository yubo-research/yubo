from __future__ import annotations

import warnings

import numpy as np

from problems.dm_control_env_core import PIXEL_HEIGHT, DMControlEnv
from problems.dm_control_spaces import (
    BoxSpace,
    DictSpace,
    is_gl_init_error,
    resize_pixels,
)


class PixelObsWrapper:
    def __init__(self, env: DMControlEnv, *, pixels_only: bool = True, size: int = 84):
        self.env = env
        self._pixels_only = bool(pixels_only)
        self._size = int(size)
        self._render_disabled = False

        pixel_space = BoxSpace(
            low=np.zeros((self._size, self._size, 3), dtype=np.uint8),
            high=np.full((self._size, self._size, 3), 255, dtype=np.uint8),
            dtype=np.uint8,
        )
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)
        if self._pixels_only:
            self.observation_space = pixel_space
        else:
            self.observation_space = DictSpace({"pixels": pixel_space, "state": env.observation_space})

    def _get_pixels(self) -> np.ndarray:
        if self._render_disabled:
            return np.zeros((self._size, self._size, 3), dtype=np.uint8)
        try:
            img = self.env.render()
        except Exception as exc:
            if not is_gl_init_error(exc):
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
        return resize_pixels(img, self._size, self._size)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        state_obs, info = self.env.reset(seed=seed, options=options)
        pixels = self._get_pixels()
        if self._pixels_only:
            return pixels, info
        return {"pixels": pixels, "state": state_obs}, info

    def step(self, action):
        state_obs, reward, terminated, truncated, info = self.env.step(action)
        pixels = self._get_pixels()
        if self._pixels_only:
            return pixels, reward, terminated, truncated, info
        return (
            {"pixels": pixels, "state": state_obs},
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        return self.env.close()


def make_dm_control(
    env_name: str,
    *,
    render_mode: str | None = None,
    from_pixels: bool = False,
    pixels_only: bool = True,
):
    from problems.dm_control_env_core import parse_env_name

    domain, task = parse_env_name(env_name)
    use_render_mode = "rgb_array" if from_pixels else render_mode
    base = DMControlEnv(domain, task, render_mode=use_render_mode)
    if from_pixels:
        return PixelObsWrapper(base, pixels_only=bool(pixels_only), size=PIXEL_HEIGHT)
    return base
