from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchrl.data as tr_data
import torchrl.modules as tr_modules

from rl.checkpointing import CheckpointManager
from rl.core import runtime as torchrl_common


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
    obs_scaler: torchrl_common.ObsScaler
    actor: tr_modules.ProbabilisticActor
    actor_model: nn.Module
    q1: nn.Module
    q2: nn.Module
    q1_target: nn.Module
    q2_target: nn.Module
    log_alpha: nn.Parameter


@dataclasses.dataclass
class _TrainingSetup:
    replay: tr_data.TensorDictReplayBuffer
    actor_optimizer: torch.optim.AdamW
    critic_optimizer: torch.optim.AdamW
    alpha_optimizer: torch.optim.AdamW
    exp_dir: Path
    metrics_path: Path
    checkpoint_manager: CheckpointManager


@dataclasses.dataclass
class _TrainState:
    start_step: int = 0
    best_return: float = -float("inf")
    best_actor_state: dict | None = None
    last_eval_return: float = float("nan")
    last_heldout_return: float | None = None
