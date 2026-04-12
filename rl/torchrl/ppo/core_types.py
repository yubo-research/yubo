from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import tensordict.nn as td_nn
import torch
import torch.nn as nn
import torch.optim as optim
import torchrl.modules.distributions as tr_dists

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


@dataclasses.dataclass
class _TrainingSetup:
    frames_per_batch: int
    num_iterations: int
    env: object | None
    loss_module: Any
    gae: Any
    train_params: list[torch.nn.Parameter]
    optimizer: optim.AdamW
    exp_dir: Any
    metrics_path: Any
    checkpoint_manager: Any


@dataclasses.dataclass
class _TrainState:
    start_iteration: int = 0
    best_return: float = -float("inf")
    best_actor_state: dict | None = None
    last_eval_return: float = float("nan")
    last_heldout_return: float | None = None


def _tanh_normal_base():
    return tr_dists.TanhNormal


class _TanhNormal(_tanh_normal_base()):
    @property
    def support(self):
        return torch.distributions.constraints.real
