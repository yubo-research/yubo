"""PPO puffer engine: thin facade for static dependency metrics."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl.pufferlib.ppo.config import PufferPPOConfig, TrainResult
    from rl.pufferlib.ppo.engine_helpers import build_eval_env_conf, make_vector_env
    from rl.pufferlib.ppo.engine_impl import register, train_ppo_puffer, train_ppo_puffer_impl

_IMPL = "rl.pufferlib.ppo.engine_impl"


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
