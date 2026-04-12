"""SAC puffer engine: thin facade."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl.pufferlib.sac.config import SACConfig, TrainResult
    from rl.pufferlib.sac.sac_puffer_engine_impl import register, train_sac_puffer, train_sac_puffer_impl

_IMPL = "rl.pufferlib.sac.sac_puffer_engine_impl"


def __getattr__(name: str):
    return getattr(importlib.import_module(_IMPL), name)


__all__ = [
    "SACConfig",
    "TrainResult",
    "register",
    "train_sac_puffer",
    "train_sac_puffer_impl",
]
