"""Runtime helpers for native Puffer SAC."""

from __future__ import annotations

import torch

from rl.core.runtime import ObsScaler as _ObsScaler
from rl.core.runtime import mps_is_available as _mps_is_available_core
from rl.core.runtime import obs_scale_from_env as _obs_scale_from_env_core
from rl.core.runtime import select_device as _select_device_core

ObsScaler = _ObsScaler


def _mps_is_available() -> bool:
    return _mps_is_available_core()


def select_device(device: str) -> torch.device:
    return _select_device_core(
        device,
        cuda_is_available_fn=torch.cuda.is_available,
        mps_is_available_fn=_mps_is_available,
    )


def obs_scale_from_env(env_conf):
    return _obs_scale_from_env_core(env_conf)
