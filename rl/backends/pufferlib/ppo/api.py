"""PufferLib PPO trainer facade.

This module keeps stable patch points used by tests and callers while delegating
implementation to ``rl.backends.pufferlib.ppo.engine``.
"""

from __future__ import annotations

from rl import registry
from rl.backends.pufferlib.ppo.config import PufferPPOConfig, TrainResult
from rl.backends.pufferlib.ppo.engine import (
    _build_eval_env_conf as _build_eval_env_conf_impl,
)
from rl.backends.pufferlib.ppo.engine import (
    _make_vector_env as _make_vector_env_impl,
)
from rl.backends.pufferlib.ppo.engine import (
    _resolve_gym_env_name as _resolve_gym_env_name_impl,
)
from rl.backends.pufferlib.ppo.engine import (
    _to_puffer_game_name as _to_puffer_game_name_impl,
)
from rl.backends.pufferlib.ppo.engine import train_ppo_puffer_impl
from rl.pufferlib_compat import import_pufferlib_modules

__all__ = [
    "PufferPPOConfig",
    "TrainResult",
    "register",
    "train_ppo_puffer",
]


def _to_puffer_game_name(env_tag: str) -> str:
    return _to_puffer_game_name_impl(env_tag)


def _resolve_gym_env_name(env_tag: str) -> tuple[str, dict]:
    return _resolve_gym_env_name_impl(env_tag)


def _make_vector_env(config: PufferPPOConfig):
    return _make_vector_env_impl(
        config,
        import_pufferlib_modules_fn=import_pufferlib_modules,
        to_puffer_game_name_fn=_to_puffer_game_name,
        resolve_gym_env_name_fn=_resolve_gym_env_name,
    )


def _build_eval_env_conf(config: PufferPPOConfig, *, obs_spec):
    return _build_eval_env_conf_impl(config, obs_spec=obs_spec)


def train_ppo_puffer(config: PufferPPOConfig) -> TrainResult:
    return train_ppo_puffer_impl(
        config,
        make_vector_env_fn=_make_vector_env,
        build_eval_env_conf_fn=_build_eval_env_conf,
    )


def register():
    registry.register_algo("ppo_puffer", PufferPPOConfig, train_ppo_puffer)
