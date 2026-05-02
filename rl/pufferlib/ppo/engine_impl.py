"""PPO puffer training implementation (imported lazily by engine facade)."""

from __future__ import annotations

import importlib


def train_ppo_puffer_impl(config):
    t = importlib.import_module("rl.pufferlib.ppo.engine_impl_train_run")
    return t.train_ppo_puffer_impl(config)


def train_ppo_puffer(config):
    t = importlib.import_module("rl.pufferlib.ppo.engine_impl_train_run")
    return t.train_ppo_puffer(config)


def register() -> None:
    reg = importlib.import_module("rl.registry")
    cfg = importlib.import_module("rl.pufferlib.ppo.config")
    reg.register_algo("ppo", cfg.PufferPPOConfig, train_ppo_puffer, backend="pufferlib")


def engine_helpers_proxy(name: str):
    _h = importlib.import_module("rl.pufferlib.ppo.engine_helpers")
    return getattr(_h, name)


def config_proxy(name: str):
    m = importlib.import_module("rl.pufferlib.ppo.config")
    return getattr(m, name)


def __getattr__(name: str):
    if name == "_make_vector_env":
        return engine_helpers_proxy("make_vector_env")
    if name in (
        "make_vector_env",
        "build_eval_env_conf",
        "_build_eval_env_conf",
        "_resolve_device",
        "_seed_everything",
        "_build_plan",
        "_prepare_outputs",
        "_init_runtime",
        "_prepare_obs",
    ):
        return engine_helpers_proxy(name)
    if name in ("PufferPPOConfig", "TrainResult"):
        return config_proxy(name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
