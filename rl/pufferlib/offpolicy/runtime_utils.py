from __future__ import annotations

import torch

from rl.core import runtime

ObsScaler = runtime.ObsScaler
_mps_is_available_core = runtime.mps_is_available


def _mps_is_available() -> bool:
    return runtime.mps_is_available()


def select_device(device: str) -> torch.device:
    return runtime.select_device(
        device,
        cuda_is_available_fn=torch.cuda.is_available,
        mps_is_available_fn=_mps_is_available,
    )


def obs_scale_from_env(env_conf):
    return runtime.obs_scale_from_env(env_conf)
