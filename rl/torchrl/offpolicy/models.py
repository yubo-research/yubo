from __future__ import annotations

import torch
import torch.nn as nn


class ActorNet(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, obs_scaler: nn.Module, act_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.obs_scaler = obs_scaler
        self.act_dim = int(act_dim)

    def forward(self, observation: torch.Tensor):
        obs = self.obs_scaler(observation)
        feats = self.backbone(obs)
        out = self.head(feats)
        loc, log_scale = (out[..., : self.act_dim], out[..., self.act_dim :])
        scale = log_scale.clamp(-5.0, 2.0).exp()
        return (loc, scale)

    def mean_log_std(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc, scale = self.forward(observation)
        return (loc, torch.log(scale))

    def log_prob_from_action(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        loc, scale = self.forward(observation)
        dist = torch.distributions.Normal(loc, scale)
        action_clamped = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = torch.atanh(action_clamped)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - action_clamped.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)

    def sample(self, obs: torch.Tensor, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        loc, scale = self.forward(obs)
        if deterministic:
            action = torch.tanh(loc)
            log_prob = torch.zeros(action.shape[0], dtype=action.dtype, device=action.device)
            return (action, log_prob)
        dist = torch.distributions.Normal(loc, scale)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        return (action, log_prob.sum(dim=-1))

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        loc, _ = self.forward(obs)
        return torch.tanh(loc)


class QNet(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, obs_scaler: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.obs_scaler = obs_scaler

    def forward(self, observation: torch.Tensor, action: torch.Tensor):
        obs = self.obs_scaler(observation)
        feats = self.backbone(torch.cat([obs, action], dim=-1))
        return self.head(feats).squeeze(-1)


class QNetPixel(nn.Module):
    def __init__(self, obs_encoder: nn.Module, head: nn.Module, obs_scaler: nn.Module):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.head = head
        self.obs_scaler = obs_scaler

    def forward(self, observation: torch.Tensor, action: torch.Tensor):
        obs = self.obs_scaler(observation)
        latent = self.obs_encoder(obs)
        return self.head(torch.cat([latent, action], dim=-1)).squeeze(-1)
