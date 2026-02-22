"""Shared Gaussian actor module for RL and BO. Uses rl/backbone for architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rl.backbone import BackboneSpec, HeadSpec, build_backbone, build_mlp_head


def _init_linear(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class SharedGaussianActorModule(nn.Module):
    """Gaussian actor built from BackboneSpec + HeadSpec."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        backbone_spec: BackboneSpec,
        head_spec: HeadSpec,
        *,
        init_log_std: float = -0.5,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
    ):
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone_spec, obs_dim)
        self.head = build_mlp_head(head_spec, feat_dim, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std), dtype=torch.float32))
        self._min_log_std = float(min_log_std)
        self._max_log_std = float(max_log_std)

        _init_linear(self.backbone)
        _init_linear(self.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)

    def _expanded_log_std(self, mean: torch.Tensor) -> torch.Tensor:
        log_std = torch.clamp(self.log_std, self._min_log_std, self._max_log_std)
        view_shape = (1,) * (mean.ndim - 1) + (mean.shape[-1],)
        return log_std.view(view_shape).expand_as(mean)

    def dist(self, x: torch.Tensor) -> Normal:
        mean = self.forward(x)
        log_std = self._expanded_log_std(mean)
        return Normal(mean, torch.exp(log_std))

    def sample_action(self, x: torch.Tensor, *, deterministic: bool = False):
        dist = self.dist(x)
        raw_action = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return raw_action, log_prob, entropy


# Registry: variant name -> (BackboneSpec, HeadSpec)
_GAUSSIAN_ACTOR_SPECS: dict[str, tuple[BackboneSpec, HeadSpec]] = {
    "rl-gauss": (
        BackboneSpec("mlp", (64, 64), "silu", layer_norm=True),
        HeadSpec(),
    ),
    "rl-gauss-tanh": (
        BackboneSpec("mlp", (16, 16), "tanh", layer_norm=False),
        HeadSpec(),
    ),
    "rl-gauss-small": (
        BackboneSpec("mlp", (32, 16), "silu", layer_norm=True),
        HeadSpec(),
    ),
}


def get_gaussian_actor_spec(variant: str) -> tuple[BackboneSpec, HeadSpec]:
    if variant not in _GAUSSIAN_ACTOR_SPECS:
        raise ValueError(f"Unknown Gaussian actor variant '{variant}'. Available: {sorted(_GAUSSIAN_ACTOR_SPECS)}")
    return _GAUSSIAN_ACTOR_SPECS[variant]


def build_shared_gaussian_actor(
    obs_dim: int,
    act_dim: int,
    variant: str = "rl-gauss",
    *,
    init_log_std: float = -0.5,
) -> SharedGaussianActorModule:
    backbone_spec, head_spec = get_gaussian_actor_spec(variant)
    return SharedGaussianActorModule(
        obs_dim,
        act_dim,
        backbone_spec,
        head_spec,
        init_log_std=init_log_std,
    )
