"""Transform pixels to observation for both root and next (for GAE bootstrap)."""

from __future__ import annotations

import torch
from torchrl.data import UnboundedContinuous
from torchrl.envs.transforms import Transform

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


def ensure_atari_obs_format(obs: torch.Tensor, size: int = 84, *, scale_float_255: bool = False) -> torch.Tensor:
    """Backward-compatible Atari wrapper for 4-channel stacked pixels."""
    return ensure_pixel_obs_format(
        obs,
        channels=4,
        size=size,
        scale_float_255=scale_float_255,
    )


class PixelsToObservation(Transform):
    """Transform pixels -> observation for root and (next, pixels) -> (next, observation)."""

    def __init__(self, size: int = 84):
        super().__init__(
            in_keys=["pixels", ("next", "pixels")],
            out_keys=["observation", ("next", "observation")],
        )
        self._size = int(size)

    def _process_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        return ensure_pixel_obs_format(
            pixels,
            channels=3,
            size=self._size,
            scale_float_255=False,
        )

    def _call(self, tensordict):
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            val = tensordict.get(in_key, None)
            if val is not None:
                tensordict.set(out_key, self._process_pixels(val))
        return tensordict

    def _reset(self, tensordict, tensordict_reset):
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            val = tensordict_reset.get(in_key, None)
            if val is not None:
                tensordict_reset.set(out_key, self._process_pixels(val))
        return tensordict_reset

    def transform_observation_spec(self, spec):
        obs_spec = UnboundedContinuous(
            shape=torch.Size((3, self._size, self._size)),
            device=spec.device,
            dtype=torch.float32,
        )
        if "pixels" in spec.keys(True, True):
            spec["observation"] = obs_spec
            # GAE needs ("next", "observation") for bootstrap; add to spec for validation
            spec[("next", "observation")] = obs_spec
        return spec


class AtariObservationTransform(Transform):
    """Transform Atari observation to (C,H,W) float for nature_cnn_atari.
    Handles (stack,H,W,1) from FrameStack or (H,W,C) from other setups."""

    def __init__(self, size: int = 84):
        super().__init__(
            in_keys=["observation", ("next", "observation")],
            out_keys=["observation", ("next", "observation")],
        )
        self._size = int(size)

    def _process(self, obs: torch.Tensor) -> torch.Tensor:
        return ensure_pixel_obs_format(
            obs,
            channels=4,
            size=self._size,
            scale_float_255=False,
        )

    def _call(self, tensordict):
        for key in self.in_keys:
            val = tensordict.get(key, None)
            if val is not None:
                tensordict.set(key, self._process(val))
        return tensordict

    def _reset(self, tensordict, tensordict_reset):
        for key in self.in_keys:
            val = tensordict_reset.get(key, None)
            if val is not None:
                tensordict_reset.set(key, self._process(val))
        return tensordict_reset

    def transform_observation_spec(self, spec):
        # Atari: (H,W,C) -> (C,H,W) with C=4 stacked frames
        obs_spec = UnboundedContinuous(
            shape=torch.Size((4, self._size, self._size)),
            device=spec.device,
            dtype=torch.float32,
        )
        if "observation" in spec.keys(True, True):
            spec["observation"] = obs_spec
            spec[("next", "observation")] = obs_spec
        return spec
