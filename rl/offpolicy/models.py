"""Q-network models for pufferlib offpolicy (SAC)."""

from __future__ import annotations

import torch
import torch.nn as nn


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
