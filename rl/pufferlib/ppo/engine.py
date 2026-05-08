"""PPO puffer engine: thin facade for static dependency metrics."""

import importlib
from typing import TYPE_CHECKING

from rl.pufferlib.ppo import engine_helpers as _helpers
from rl.pufferlib.ppo.config import PufferPPOConfig

if TYPE_CHECKING:
    from rl.pufferlib.ppo.config import PufferPPOConfig, TrainResult
    from rl.pufferlib.ppo.engine_helpers import build_eval_env_conf, make_vector_env
    from rl.pufferlib.ppo.engine_impl import (
        register,
        train_ppo_puffer,
        train_ppo_puffer_impl,
    )

_IMPL = "rl.pufferlib.ppo.engine_impl"
_make_vector_env = _helpers.make_vector_env
_build_eval_env_conf = _helpers.build_eval_env_conf


def make_vector_env(config):
    return _make_vector_env(config)


def build_eval_env_conf(config, obs_spec):
    return _build_eval_env_conf(config, obs_spec=obs_spec)


def train_ppo_puffer_impl(config):
    return importlib.import_module(_IMPL).train_ppo_puffer_impl(config)


def train_ppo_puffer(config):
    return train_ppo_puffer_impl(config)


def register() -> None:
    reg = importlib.import_module("rl.registry")
    reg.register_algo("ppo", PufferPPOConfig, train_ppo_puffer, backend="pufferlib")


def __getattr__(name: str):
    return getattr(importlib.import_module(_IMPL), name)


__all__ = [
    "PufferPPOConfig",
    "TrainResult",
    "build_eval_env_conf",
    "make_vector_env",
    "register",
    "train_ppo_puffer",
    "train_ppo_puffer_impl",
]
