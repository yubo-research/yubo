"""Continuous action-space helpers shared across RL algorithms/backends."""

from __future__ import annotations

import numpy as np
import torch


def normalize_action_bounds(low: np.ndarray, high: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    low_v = np.asarray(low, dtype=np.float32).reshape(-1)
    high_v = np.asarray(high, dtype=np.float32).reshape(-1)
    if low_v.size == 1 and dim > 1:
        low_v = np.full((dim,), float(low_v.item()), dtype=np.float32)
    if high_v.size == 1 and dim > 1:
        high_v = np.full((dim,), float(high_v.item()), dtype=np.float32)
    if low_v.size != dim or high_v.size != dim:
        raise ValueError(f"Action bounds must match action dimension {dim}: low={low_v.shape}, high={high_v.shape}")

    low_v = np.where(np.isfinite(low_v), low_v, -1.0)
    high_v = np.where(np.isfinite(high_v), high_v, 1.0)

    # Treat effectively-unbounded Box limits as normalized control range [-1, 1].
    f32_max = float(np.finfo(np.float32).max)
    sentinel = (np.abs(low_v) >= 0.5 * f32_max) | (np.abs(high_v) >= 0.5 * f32_max)
    low_v = np.where(sentinel, -1.0, low_v)
    high_v = np.where(sentinel, 1.0, high_v)

    width = high_v - low_v
    bad_width = (~np.isfinite(width)) | (width <= 0.0)
    low_v = np.where(bad_width, -1.0, low_v)
    high_v = np.where(bad_width, 1.0, high_v)

    high_v = np.maximum(high_v, low_v + 1e-6)
    return low_v, high_v


def scale_action_to_env(action: np.ndarray, low: np.ndarray, high: np.ndarray, *, clip: bool = True) -> np.ndarray:
    mapped = np.asarray(low, dtype=np.float32) + 0.5 * (np.asarray(action, dtype=np.float32) + 1.0) * (
        np.asarray(high, dtype=np.float32) - np.asarray(low, dtype=np.float32)
    )
    if clip:
        return np.clip(mapped, np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32))
    return mapped


def unscale_action_from_env(action: np.ndarray, low: np.ndarray, high: np.ndarray, *, clip: bool = True) -> np.ndarray:
    low_v = np.asarray(low, dtype=np.float32)
    high_v = np.asarray(high, dtype=np.float32)
    width = np.maximum(high_v - low_v, 1e-8)
    scaled = 2.0 * (np.asarray(action, dtype=np.float32) - low_v) / width - 1.0
    if clip:
        return np.clip(scaled, -1.0, 1.0)
    return scaled


def unscale_action_tensor_from_env(
    action: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    clip: bool = True,
) -> torch.Tensor:
    width = (high - low).clamp(min=1e-8)
    scaled = 2.0 * (action - low) / width - 1.0
    if clip:
        return scaled.clamp(-1.0, 1.0)
    return scaled


def scale_action_tensor_to_env(
    action: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    clip: bool = True,
) -> torch.Tensor:
    mapped = low + 0.5 * (action + 1.0) * (high - low)
    if clip:
        return torch.max(torch.min(mapped, high), low)
    return mapped
