"""Core data/model specs for pufferlib PPO."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.categorical import Categorical


@dataclasses.dataclass(frozen=True)
class _ObservationSpec:
    mode: str  # vector | pixels
    raw_shape: tuple[int, ...]
    vector_dim: int | None = None
    channels: int | None = None
    image_size: int | None = None


@dataclasses.dataclass(frozen=True)
class _ActionSpec:
    kind: str  # discrete | continuous
    dim: int
    low: np.ndarray | None = None
    high: np.ndarray | None = None


class _ActorCritic(nn.Module):
    def __init__(
        self,
        *,
        actor_backbone: nn.Module,
        critic_backbone: nn.Module,
        actor_head: nn.Module,
        critic_head: nn.Module,
        action_spec: _ActionSpec,
        log_std_init: float,
    ):
        super().__init__()
        self.actor_backbone = actor_backbone
        self.critic_backbone = critic_backbone
        self.actor_head = actor_head
        self.critic_head = critic_head
        self.action_spec = action_spec
        self._action_kind = str(action_spec.kind)

        if self._action_kind == "continuous":
            dim = int(action_spec.dim)
            low = np.asarray(action_spec.low, dtype=np.float32).reshape(-1)
            high = np.asarray(action_spec.high, dtype=np.float32).reshape(-1)
            if low.size != dim or high.size != dim:
                raise ValueError("Continuous action bounds must match action dimension.")
            self.log_std = nn.Parameter(torch.full((dim,), float(log_std_init), dtype=torch.float32))
            self.register_buffer(
                "action_low",
                torch.as_tensor(low, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "action_high",
                torch.as_tensor(high, dtype=torch.float32),
                persistent=False,
            )

    def _actor_features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_backbone(obs)

    def _critic_features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_backbone(obs)

    def _expand_log_std(self, mean: torch.Tensor) -> torch.Tensor:
        log_std = torch.clamp(self.log_std, -20.0, 2.0)
        view_shape = (1,) * (mean.ndim - 1) + (mean.shape[-1],)
        return log_std.view(view_shape).expand_as(mean)

    def _clip_action(self, action: torch.Tensor) -> torch.Tensor:
        if self._action_kind != "continuous":
            return action
        view_shape = (1,) * (action.ndim - 1) + (action.shape[-1],)
        low = self.action_low.view(view_shape)
        high = self.action_high.view(view_shape)
        return torch.max(torch.min(action, high), low)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actor_out = self.actor_head(self._actor_features(obs))
        value = self.critic_head(self._critic_features(obs)).squeeze(-1)
        return actor_out, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)[1]

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        actor_out, value = self.forward(obs)
        if self._action_kind == "discrete":
            logits = actor_out
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            return action, dist.log_prob(action), dist.entropy(), value

        mean = actor_out
        std = torch.exp(self._expand_log_std(mean))
        dist = Normal(mean, std)
        if action is None:
            action = dist.rsample()
        action = self._clip_action(action)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value


@dataclasses.dataclass(frozen=True)
class _TrainPlan:
    num_envs: int
    num_steps: int
    batch_size: int
    minibatch_size: int
    num_iterations: int


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


@dataclasses.dataclass
class _UpdateStats:
    approx_kl: float
    clipfrac_mean: float


def init_linear(module: nn.Module, gain: float) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
