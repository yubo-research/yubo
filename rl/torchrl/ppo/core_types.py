from __future__ import annotations

import dataclasses
from typing import Any

import torch
import torch.optim as optim
import torchrl.modules.distributions as tr_dists

from .core_types_env import (  # noqa: F401
    _EnvSetup,
    _Modules,
)


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
