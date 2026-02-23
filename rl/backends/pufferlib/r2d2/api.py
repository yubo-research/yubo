"""PufferLib R2D2 backend API."""

from __future__ import annotations

from rl import registry
from rl.backends.pufferlib.r2d2.config import R2D2Config
from rl.backends.pufferlib.r2d2.engine import TrainResult, train_r2d2

__all__ = [
    "R2D2Config",
    "TrainResult",
    "register",
    "train_r2d2",
]


def register() -> None:
    registry.register_algo("r2d2", R2D2Config, train_r2d2)
