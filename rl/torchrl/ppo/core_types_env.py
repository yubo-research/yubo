from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import tensordict.nn as td_nn
import torch.nn as nn

import rl.core.env_contract as torchrl_env_contract


@dataclasses.dataclass(frozen=True)
class _EnvSetup:
    env_conf: object
    io_contract: torchrl_env_contract.EnvIOContract
    problem_seed: int
    noise_seed_0: int
    obs_dim: int
    act_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    obs_lb: np.ndarray | None
    obs_width: np.ndarray | None
    is_discrete: bool = False


@dataclasses.dataclass
class _Modules:
    actor_backbone: nn.Module
    actor_head: nn.Module
    critic_backbone: nn.Module
    critic_head: nn.Module
    log_std: nn.Parameter | None
    obs_scaler: Any
    actor: Any
    critic: td_nn.TensorDictModule
