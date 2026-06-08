from __future__ import annotations

import dataclasses

import numpy as np
import torch.nn as nn
import torchrl.modules as tr_modules

from rl.core import runtime


@dataclasses.dataclass(frozen=True)
class _EnvSetup:
    env_conf: object
    problem_seed: int
    noise_seed_0: int
    obs_dim: int
    act_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    obs_lb: np.ndarray | None
    obs_width: np.ndarray | None


@dataclasses.dataclass
class _Modules:
    actor_backbone: nn.Module
    actor_head: nn.Module
    obs_scaler: runtime.ObsScaler
    actor: tr_modules.ProbabilisticActor
    actor_model: nn.Module
    q1: nn.Module
    q2: nn.Module
    q1_target: nn.Module
    q2_target: nn.Module
    log_alpha: nn.Parameter
