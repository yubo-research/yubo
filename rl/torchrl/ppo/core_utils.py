from __future__ import annotations

import torch
import torch.nn as nn

from rl.core import env_contract

from .config import PPOConfig
from .core_types import _EnvSetup


def _unique_params_by_id(*modules: nn.Module, extra_params: list[torch.nn.Parameter] | None = None) -> dict[int, torch.nn.Parameter]:
    unique: dict[int, torch.nn.Parameter] = {}
    for module in modules:
        for p in module.parameters():
            unique[id(p)] = p
    if extra_params:
        for p in extra_params:
            unique[id(p)] = p
    return unique


def _count_unique_params(*modules: nn.Module, extra_params: list[torch.nn.Parameter] | None = None) -> int:
    unique = _unique_params_by_id(*modules, extra_params=extra_params)
    return sum((p.numel() for p in unique.values()))


def _unique_param_list(*modules: nn.Module, extra_params: list[torch.nn.Parameter] | None = None) -> list[torch.nn.Parameter]:
    return list(_unique_params_by_id(*modules, extra_params=extra_params).values())


def _is_due(step: int, interval: int | None) -> bool:
    return interval is not None and int(interval) > 0 and (int(step) % int(interval) == 0)


def _is_dm_control_env(env_conf) -> bool:
    return getattr(env_conf, "env_name", "").startswith("dm_control/")


def _is_atari_env(env_conf) -> bool:
    return getattr(env_conf, "env_name", "").startswith("ALE/")


def _resolve_observation_contract_for_env(config: PPOConfig, env: _EnvSetup | object) -> env_contract.ObservationContract:
    io_contract = getattr(env, "io_contract", None)
    observation = getattr(io_contract, "observation", None)
    if observation is not None:
        return observation
    env_conf = getattr(env, "env_conf", None)
    if env_conf is not None:
        return env_contract.resolve_observation_contract(env_conf, default_image_size=84)
    if bool(getattr(config, "from_pixels", False)):
        return env_contract.ObservationContract(mode="pixels", raw_shape=(), model_channels=3, image_size=84)
    return env_contract.ObservationContract(mode="vector", raw_shape=(), vector_dim=1)
