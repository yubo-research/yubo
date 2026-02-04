from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from rl.backbone import BackboneSpec, HeadSpec, build_backbone, build_mlp_head


def _atanh(x, eps: float = 1e-6):
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _init_linear(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


@dataclass
class PolicySpecs:
    backbone: BackboneSpec
    actor_head: HeadSpec
    critic_head: HeadSpec
    share_backbone: bool = True
    log_std_init: float = 0.0


class ActionValue(NamedTuple):
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        specs: PolicySpecs,
        *,
        actor_backbone: BackboneSpec | None = None,
        critic_backbone: BackboneSpec | None = None,
    ):
        super().__init__()
        obs_dim = int(obs_dim)
        act_dim = int(act_dim)

        if actor_backbone is None:
            actor_backbone = specs.backbone
        if critic_backbone is None:
            critic_backbone = specs.backbone

        if specs.share_backbone:
            backbone, feat_dim = build_backbone(actor_backbone, obs_dim)
            self.actor_backbone = backbone
            self.critic_backbone = backbone
            actor_feat_dim = feat_dim
            critic_feat_dim = feat_dim
        else:
            self.actor_backbone, actor_feat_dim = build_backbone(actor_backbone, obs_dim)
            self.critic_backbone, critic_feat_dim = build_backbone(critic_backbone, obs_dim)

        self.actor_head = build_mlp_head(specs.actor_head, actor_feat_dim, act_dim)
        self.critic_head = build_mlp_head(specs.critic_head, critic_feat_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(specs.log_std_init)))

        _init_linear(self.actor_backbone)
        if not specs.share_backbone:
            _init_linear(self.critic_backbone)
        _init_linear(self.actor_head)
        _init_linear(self.critic_head)

    def actor_num_params(self) -> int:
        params = list(self.actor_backbone.parameters()) + list(self.actor_head.parameters()) + [self.log_std]
        return sum(p.numel() for p in params)

    def _distribution(self, obs: torch.Tensor):
        feats = self.actor_backbone(obs)
        mean = self.actor_head(feats)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        feats = self.critic_backbone(obs)
        return self.critic_head(feats).squeeze(-1)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self._distribution(obs)
        mean = dist.mean
        return torch.tanh(mean)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        dist = self._distribution(obs)
        if action is None:
            u = dist.rsample()
            action = torch.tanh(u)
        else:
            u = _atanh(action)
        log_prob = dist.log_prob(u) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.get_value(obs)
        return ActionValue(action=action, log_prob=log_prob, entropy=entropy, value=value)
