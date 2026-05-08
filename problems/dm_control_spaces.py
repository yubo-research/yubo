from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import zoom


class BoxSpace:
    def __init__(self, low: np.ndarray, high: np.ndarray, dtype=np.float32):
        self.low = np.asarray(low, dtype=dtype)
        self.high = np.asarray(high, dtype=dtype)
        self.dtype = np.dtype(dtype)
        self.shape = tuple(int(v) for v in self.low.shape)

    def sample(self):
        finite = np.isfinite(self.low) & np.isfinite(self.high)
        out = np.zeros(self.shape, dtype=self.dtype)
        if np.any(finite):
            out[finite] = np.random.uniform(self.low[finite], self.high[finite]).astype(self.dtype)
        if np.any(~finite):
            out[~finite] = np.random.normal(size=int(np.sum(~finite))).astype(self.dtype)
        return out


class DictSpace:
    def __init__(self, spaces: dict[str, Any]):
        self.spaces = dict(spaces)

    def sample(self):
        return {k: v.sample() for k, v in self.spaces.items()}


def spec_bounds(spec: Any) -> tuple[np.ndarray, np.ndarray]:
    minimum = getattr(spec, "minimum", None)
    maximum = getattr(spec, "maximum", None)
    shape = spec.shape
    low = np.full(shape, -np.inf, dtype=np.float32)
    high = np.full(shape, np.inf, dtype=np.float32)
    if minimum is not None:
        low = np.asarray(minimum, dtype=np.float32)
    if maximum is not None:
        high = np.asarray(maximum, dtype=np.float32)
    return low, high


def flatten_obs(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        parts = [np.ravel(np.asarray(obs[k], dtype=np.float32)) for k in sorted(obs)]
        if not parts:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(parts, axis=0)
    return np.ravel(np.asarray(obs, dtype=np.float32))


def spec_to_space(spec: Any):
    if isinstance(spec, dict):
        lows, highs = [], []
        for key in sorted(spec):
            low, high = spec_bounds(spec[key])
            lows.append(np.ravel(low))
            highs.append(np.ravel(high))
        low = np.concatenate(lows, axis=0)
        high = np.concatenate(highs, axis=0)
        return BoxSpace(low=low, high=high, dtype=np.float32)
    low, high = spec_bounds(spec)
    return BoxSpace(low=low, high=high, dtype=np.float32)


def resize_pixels(pixels: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = pixels.shape[:2]
    if h == target_h and w == target_w:
        return np.asarray(pixels, dtype=np.uint8)
    factors = (target_h / h, target_w / w, 1.0)
    out = zoom(pixels, factors, order=0)
    return np.asarray(out, dtype=np.uint8)


def is_gl_init_error(exc: Exception) -> bool:
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
