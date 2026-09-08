from __future__ import annotations

import dataclasses

import common.env_preprocessing as env_pre
from problems.env_conf import get_env_conf
from rl.core import env_contract, env_setup, runtime

from .config import PPOConfig
from .core_types import _EnvSetup


def build_env_setup(config: PPOConfig) -> _EnvSetup:
    resolved = env_setup.build_env_setup(
        env_tag=str(config.env_tag),
        seed=int(config.seed),
        problem_seed=config.problem_seed,
        noise_seed_0=config.noise_seed_0,
        from_pixels=bool(getattr(config, "from_pixels", False)),
        pixels_only=bool(getattr(config, "pixels_only", True)),
        get_env_conf_fn=get_env_conf,
    )
    env_conf = resolved.env_conf
    env_pre.attach_env_preprocessing(env_conf, _gym_preprocessing_config(config))
    env_conf.ensure_spaces()
    io_contract = env_contract.resolve_env_io_contract(env_conf, default_image_size=84)
    obs_dim = 64 if io_contract.observation.mode == "pixels" else int(io_contract.observation.vector_dim or 1)
    act_dim = int(io_contract.action.dim)
    action_low = io_contract.action.low
    action_high = io_contract.action.high
    lb, width = runtime.obs_scale_from_env(env_conf)
    return _EnvSetup(
        env_conf=env_conf,
        io_contract=io_contract,
        problem_seed=int(resolved.problem_seed),
        noise_seed_0=int(resolved.noise_seed_0),
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low=action_low,
        action_high=action_high,
        obs_lb=lb,
        obs_width=width,
        is_discrete=io_contract.action.kind == "discrete",
    )


def _build_seeded_eval_env_conf(config: PPOConfig, *, problem_seed: int, noise_seed_0: int, from_pixels: bool):
    resolved = env_setup.build_env_setup(
        env_tag=str(config.env_tag),
        seed=int(config.seed),
        problem_seed=int(problem_seed),
        noise_seed_0=int(noise_seed_0),
        from_pixels=bool(from_pixels),
        pixels_only=bool(getattr(config, "pixels_only", True)),
        get_env_conf_fn=get_env_conf,
    )
    return env_pre.attach_env_preprocessing(resolved.env_conf, _gym_preprocessing_config(config))


def _gym_preprocessing_config(config: PPOConfig):
    return dataclasses.replace(config.env, normalize_reward=False)


def _build_eval_env_conf(config: PPOConfig, env: _EnvSetup, *, from_pixels: bool):
    return _build_seeded_eval_env_conf(
        config,
        problem_seed=int(env.problem_seed),
        noise_seed_0=int(env.noise_seed_0),
        from_pixels=from_pixels,
    )
