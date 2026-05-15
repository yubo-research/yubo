from __future__ import annotations

import importlib
from typing import Any

from . import eval_utils_impl  # noqa: F401
from .types import SACConfig, TrainResult

_EXPORTS: dict[str, tuple[str, str]] = {
    "ReplayBuffer": ("rl.pufferlib.sac.replay", "ReplayBuffer"),
    "register": ("rl.pufferlib.sac.engine", "register"),
    "train_sac_puffer": ("rl.pufferlib.sac.engine", "train_sac_puffer"),
    "train_sac_puffer_impl": ("rl.pufferlib.sac.engine", "train_sac_puffer_impl"),
}


def __getattr__(name: str) -> Any:
    spec = _EXPORTS.get(name)
    if spec is None:
        raise AttributeError(name)
    module_name, attr = spec
    return getattr(importlib.import_module(module_name), attr)


__all__ = [
    "SACConfig",
    "TrainResult",
    "ReplayBuffer",
    "register",
    "train_sac_puffer",
    "train_sac_puffer_impl",
]
