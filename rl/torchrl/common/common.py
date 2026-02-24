from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class ObsScaler(nn.Module):
    def __init__(self, lb: np.ndarray | None, width: np.ndarray | None):
        super().__init__()
        # Keep an always-registered anchor so we have a canonical dtype/device
        # even when observation scaling is disabled.
        self.register_buffer("_anchor", torch.zeros((), dtype=torch.float32))
        if lb is None or width is None:
            self._lb = None
            self._width = None
            return
        self.register_buffer("_lb", torch.as_tensor(lb, dtype=torch.float32))
        self.register_buffer("_width", torch.as_tensor(width, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self._lb is not None and self._width is not None:
            target_device = self._lb.device
            target_dtype = self._lb.dtype
        else:
            target_device = self._anchor.device
            target_dtype = self._anchor.dtype
        if obs.device != target_device:
            raise RuntimeError(f"ObsScaler expected observation on device {target_device}, got {obs.device}.")
        if obs.dtype != target_dtype:
            raise RuntimeError(
                f"ObsScaler expected observation dtype {target_dtype}, got {obs.dtype}. Ensure env/collector observation pipeline emits float32."
            )
        if self._lb is None or self._width is None:
            return obs
        return (obs - self._lb) / self._width


def _mps_is_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def select_device(device: str) -> torch.device:
    requested = str(device).strip().lower()
    if requested in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("device='cuda' requested but CUDA is not available.")
        return torch.device("cuda")
    if requested == "mps":
        if not _mps_is_available():
            raise ValueError("device='mps' requested but MPS is not available.")
        return torch.device("mps")
    raise ValueError(f"Unsupported device '{device}'. Use one of: auto, cpu, cuda, mps.")


def collector_device_kwargs(policy_device: torch.device) -> dict[str, torch.device]:
    # Gym environments step on CPU; explicit routing avoids implicit cross-device behavior.
    return {
        "env_device": torch.device("cpu"),
        "policy_device": policy_device,
        "storing_device": policy_device,
    }


def obs_scale_from_env(env_conf: Any):
    if env_conf.gym_conf is None or not env_conf.gym_conf.transform_state:
        return None, None
    env_conf.ensure_spaces()
    lb = np.asarray(env_conf.gym_conf.state_space.low, dtype=np.float32)
    width = np.asarray(
        env_conf.gym_conf.state_space.high - env_conf.gym_conf.state_space.low,
        dtype=np.float32,
    )
    if np.all(np.isinf(lb)):
        lb = np.zeros(env_conf.gym_conf.state_space.shape, dtype=np.float32)
    if np.all(np.isinf(width)):
        width = np.ones(env_conf.gym_conf.state_space.shape, dtype=np.float32)
    if np.any(np.isinf(lb)) or np.any(np.isinf(width)):
        raise ValueError("Observation bounds must be finite for transform_state.")
    return lb, width


@contextmanager
def temporary_distribution_validate_args(enabled: bool):
    previous_validate_args = getattr(torch.distributions.Distribution, "_validate_args", True)
    torch.distributions.Distribution.set_default_validate_args(bool(enabled))
    try:
        yield
    finally:
        torch.distributions.Distribution.set_default_validate_args(previous_validate_args)
