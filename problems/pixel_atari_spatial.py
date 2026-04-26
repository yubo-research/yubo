from __future__ import annotations

import torch


def atari_in_channels_from_obs_shape(shape: tuple[int, ...], *, require_spatial_84: bool) -> int:
    if len(shape) == 4 and shape[-1] == 1:
        in_channels = int(shape[0])
    elif len(shape) == 3:
        in_channels = int(shape[-1]) if shape[0] != 4 else 4
    else:
        if require_spatial_84:
            msg = f"Expected 3D or 4D obs (4,84,84), (84,84,4), (4,84,84,1), got {shape}"
        else:
            msg = f"Expected 3D or 4D obs, got {shape}"
        raise ValueError(msg)
    if require_spatial_84:
        assert 84 in shape[:3], f"Expected 84x84, got {shape}"
    return in_channels


def atari_obs_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() == 4 and x.shape[-1] == 1:
        x = x.unsqueeze(0)[..., 0]
    elif x.dim() == 5 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    if x.shape[1] in (3, 4) and x.shape[2] == 84:
        pass
    elif x.shape[-1] in (3, 4):
        x = x.permute(0, 3, 1, 2)
    return x
