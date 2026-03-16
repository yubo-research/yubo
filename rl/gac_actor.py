"""Geometric Action Control (GAC) actor for SAC.

GAC replaces Gaussian policies with spherical action generation:
  a = r * normalize(w(κ)*μ + (1-w(κ))*ξ)
where μ is the direction (unit vector), κ is concentration, ξ ~ Uniform(S^{d-1}).
See: Beyond Distributions: Geometric Action Control (Lin, ICLR 2026).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.offpolicy.runtime_utils import ObsScaler


def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    return x / norm


class GACActorNet(nn.Module):
    """GAC actor: direction + concentration heads, spherical mixing."""

    def __init__(
        self,
        backbone: nn.Module,
        direction_head: nn.Module,
        concentration_head: nn.Module,
        obs_scaler: ObsScaler,
        *,
        act_dim: int,
        action_radius: float = 2.5,
        use_adaptive_scale: bool = False,
        scale_head: nn.Module | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.direction_head = direction_head
        self.concentration_head = concentration_head
        self.obs_scaler = obs_scaler
        self.act_dim = int(act_dim)
        self.action_radius = float(action_radius)
        self.use_adaptive_scale = bool(use_adaptive_scale)
        self.scale_head = scale_head

    def _feats(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.obs_scaler(obs))

    def _direction(self, feats: torch.Tensor) -> torch.Tensor:
        raw = self.direction_head(feats)
        return _normalize(raw)

    def _concentration(self, feats: torch.Tensor) -> torch.Tensor:
        """Raw κ for w(κ)=sigmoid(κ). Paper uses unbounded κ; we clamp only in loss/target."""
        return self.concentration_head(feats).squeeze(-1)

    def _scale(self, feats: torch.Tensor) -> torch.Tensor | None:
        if self.scale_head is None:
            return None
        return F.softplus(self.scale_head(feats)) + 1.0

    def _sample_action(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self._feats(obs)
        mu = self._direction(feats)
        kappa = self._concentration(feats)
        w = torch.sigmoid(kappa)

        if deterministic:
            v = mu
        else:
            xi = torch.randn_like(mu, device=mu.device, dtype=mu.dtype)
            xi = _normalize(xi)
            v = w.unsqueeze(-1) * mu + (1 - w).unsqueeze(-1) * xi
            v = _normalize(v)

        if self.use_adaptive_scale and self.scale_head is not None:
            r = self._scale(feats)
            action = r * v
        else:
            action = self.action_radius * v

        action = action.clamp(-1.0, 1.0)
        return action, kappa

    def sample(self, obs: torch.Tensor, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action, kappa). Kappa replaces log_prob for GAC target/exploration."""
        return self._sample_action(obs, deterministic=deterministic)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        action, _ = self._sample_action(obs, deterministic=True)
        return action
