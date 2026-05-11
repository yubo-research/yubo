from __future__ import annotations

import importlib
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from rl.pufferlib.sac.config import SACConfig, TrainResult

__all__ = [
    "SACConfig",
    "TrainResult",
    "register",
    "train_sac_puffer",
    "train_sac_puffer_impl",
]


def train_sac_puffer_impl(config):
    t = importlib.import_module("rl.pufferlib.sac.sac_puffer_train_run")
    return t.train_sac_puffer_impl(config)


def train_sac_puffer(config):
    t = importlib.import_module("rl.pufferlib.sac.sac_puffer_train_run")
    return t.train_sac_puffer(config)


def register() -> None:
    reg = importlib.import_module("rl.registry")
    cfg = importlib.import_module("rl.pufferlib.sac.config")
    reg.register_algo("sac", cfg.SACConfig, train_sac_puffer, backend="pufferlib")


def config_proxy(name: str):
    m = importlib.import_module("rl.pufferlib.sac.config")
    return getattr(m, name)


def __getattr__(name: str):
    if name in ("SACConfig", "TrainResult"):
        return config_proxy(name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
