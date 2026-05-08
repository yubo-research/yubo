from __future__ import annotations

import dataclasses
from typing import Any

import torch

from .specs_dataclasses_basic import _ActionSpec, _ObservationSpec


@dataclasses.dataclass
class _RolloutBuffer:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor


@dataclasses.dataclass
class _FlatBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


@dataclasses.dataclass
class _RuntimeState:
    next_obs: torch.Tensor
    next_done: torch.Tensor
    obs_spec: _ObservationSpec
    action_spec: _ActionSpec
    global_step: int
    start_iteration: int
    start_time: float
    best_actor_state: dict | None
    best_return: float
    last_eval_return: float
    last_heldout_return: float | None
    last_episode_return: float
    eval_env_conf: Any | None
