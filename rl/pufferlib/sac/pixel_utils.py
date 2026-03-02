"""Pixel preprocessing helpers for native Puffer SAC."""

from __future__ import annotations

import torch

_INT_IMAGE_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _to_float_image(obs: torch.Tensor, *, scale_float_255: bool) -> torch.Tensor:
    if obs.dtype in _INT_IMAGE_DTYPES:
        return obs.float() / 255.0
    obs = obs.float()
    if not scale_float_255 or obs.numel() == 0:
        return obs
    min_val = float(obs.min().item())
    max_val = float(obs.max().item())
    if min_val >= 0.0 and max_val > 1.0 and max_val <= 255.0:
        return obs / 255.0
    return obs


def _canonicalize_pixel_layout(obs: torch.Tensor, *, target_channels: int, size: int) -> torch.Tensor:
    if obs.shape[-3] in (1, target_channels):
        return obs
    if obs.shape[-1] in (1, target_channels):
        return obs.movedim(-1, -3)
    if obs.ndim >= 4 and obs.shape[-2] in (1, target_channels):
        return obs.movedim(-2, -3)
    if obs.ndim == 3 and obs.shape[-2] == size and obs.shape[-1] == size:
        return obs.unsqueeze(1)
    raise ValueError(f"Unable to infer pixel channel dimension from shape {tuple(obs.shape)}")


def _match_pixel_channels(obs: torch.Tensor, *, target_channels: int) -> torch.Tensor:
    in_channels = int(obs.shape[-3])
    if in_channels == 1 and target_channels > 1:
        return obs.expand(*obs.shape[:-3], target_channels, *obs.shape[-2:])
    if in_channels != target_channels:
        raise ValueError(f"Pixel obs expected channel count {target_channels}, got {in_channels} for shape {tuple(obs.shape)}")
    return obs


def ensure_pixel_obs_format(
    obs: torch.Tensor,
    *,
    channels: int,
    size: int = 84,
    scale_float_255: bool = False,
) -> torch.Tensor:
    """Normalize pixel observations to (..., channels, size, size) float32."""
    if obs.ndim == 2:
        obs = obs.unsqueeze(0)
    if obs.ndim < 3:
        raise ValueError(f"Pixel obs expected ndim>=3, got shape {tuple(obs.shape)}")

    obs = _to_float_image(obs, scale_float_255=scale_float_255)
    if obs.ndim >= 4 and obs.shape[-1] == 1:
        obs = obs.squeeze(-1)

    target_channels = int(channels)
    obs = _canonicalize_pixel_layout(obs, target_channels=target_channels, size=size)
    obs = _match_pixel_channels(obs, target_channels=target_channels)

    if obs.shape[-2] != size or obs.shape[-1] != size:
        flat = obs.reshape(-1, int(obs.shape[-3]), int(obs.shape[-2]), int(obs.shape[-1]))
        out = torch.nn.functional.interpolate(flat, size=(size, size), mode="nearest")
        obs = out.reshape(*obs.shape[:-3], int(obs.shape[-3]), size, size)
    return obs.float()
