from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


class ObsScaler(nn.Module):
    def __init__(self, lb: np.ndarray | None, width: np.ndarray | None):
        super().__init__()
        if lb is None or width is None:
            self._lb = None
            self._width = None
            return
        self.register_buffer("_lb", torch.as_tensor(lb, dtype=torch.float32))
        self.register_buffer("_width", torch.as_tensor(width, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self._lb is None or self._width is None:
            return obs
        return (obs - self._lb) / self._width


def select_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda")
    if device == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
    width = np.asarray(env_conf.gym_conf.state_space.high - env_conf.gym_conf.state_space.low, dtype=np.float32)
    if np.all(np.isinf(lb)):
        lb = np.zeros(env_conf.gym_conf.state_space.shape, dtype=np.float32)
    if np.all(np.isinf(width)):
        width = np.ones(env_conf.gym_conf.state_space.shape, dtype=np.float32)
    if np.any(np.isinf(lb)) or np.any(np.isinf(width)):
        raise ValueError("Observation bounds must be finite for transform_state.")
    return lb, width
