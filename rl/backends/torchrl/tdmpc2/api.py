"""TorchRL TD-MPC2 backend API."""

from __future__ import annotations

from rl import registry
from rl.backends.torchrl.tdmpc2.config import TDMPC2Config
from rl.backends.torchrl.tdmpc2.trainer import TrainResult, train_tdmpc2

__all__ = [
    "TDMPC2Config",
    "TrainResult",
    "register",
    "train_tdmpc2",
]


def register() -> None:
    registry.register_algo("tdmpc2", TDMPC2Config, train_tdmpc2)
