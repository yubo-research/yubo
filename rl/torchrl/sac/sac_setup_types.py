from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
import torchrl.data as tr_data

from rl.checkpointing import CheckpointManager

from .sac_setup_env import (  # noqa: F401
    _EnvSetup,
    _Modules,
)


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
