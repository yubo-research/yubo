from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..common.env_contract import ObservationContract
from ..common.pixel_transform import ensure_pixel_obs_format


def _flatten_pixel_batch(
    obs: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...] | None]:
    if obs.ndim == 5:
        batch_shape = obs.shape[:2]
        obs = obs.flatten(0, 1)
        return obs, batch_shape
    return obs, None


def prepare_obs_for_backbone(
    obs: torch.Tensor,
    obs_contract: ObservationContract,
) -> tuple[torch.Tensor, tuple[int, ...] | None, bool]:
    if obs_contract.mode != "pixels":
        return obs, None, False

    obs, batch_shape = _flatten_pixel_batch(obs)
    obs = ensure_pixel_obs_format(
        obs,
        channels=int(obs_contract.model_channels or 3),
        size=int(obs_contract.image_size or 84),
    )
    squeeze_batch_dim = False
    if obs.ndim == 3:
        obs = obs.unsqueeze(0)
        squeeze_batch_dim = True
    return obs, batch_shape, squeeze_batch_dim


class ActorNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        log_std: torch.nn.Parameter,
        obs_scaler: Any,
        *,
        obs_contract: ObservationContract,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.log_std = log_std
        self.obs_scaler = obs_scaler
        self.obs_contract = obs_contract

    def forward(self, obs: torch.Tensor):
        obs = self.obs_scaler(obs)
        obs, batch_shape, squeeze_batch_dim = prepare_obs_for_backbone(obs, self.obs_contract)
        feats = self.backbone(obs)
        loc = self.head(feats)
        scale = self.log_std.exp().expand_as(loc)
        if batch_shape is not None:
            loc = loc.reshape(*batch_shape, -1)
            scale = scale.reshape(*batch_shape, -1)
        elif squeeze_batch_dim:
            loc = loc.squeeze(0)
            scale = scale.squeeze(0)
        return loc, scale


class DiscreteActorNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        obs_scaler: Any,
        *,
        obs_contract: ObservationContract,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.obs_scaler = obs_scaler
        self.obs_contract = obs_contract

    def forward(self, obs: torch.Tensor):
        obs = self.obs_scaler(obs)
        obs, batch_shape, squeeze_batch_dim = prepare_obs_for_backbone(obs, self.obs_contract)
        feats = self.backbone(obs)
        logits = self.head(feats)
        if batch_shape is not None:
            logits = logits.reshape(*batch_shape, -1)
        elif squeeze_batch_dim:
            logits = logits.squeeze(0)
        return logits


class CriticNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        obs_scaler: Any,
        *,
        obs_contract: ObservationContract,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.obs_scaler = obs_scaler
        self.obs_contract = obs_contract

    def forward(self, obs: torch.Tensor):
        obs = self.obs_scaler(obs)
        obs, batch_shape, squeeze_batch_dim = prepare_obs_for_backbone(obs, self.obs_contract)
        feats = self.backbone(obs)
        out = self.head(feats)
        if batch_shape is not None:
            out = out.reshape(*batch_shape, -1)
        elif squeeze_batch_dim:
            out = out.squeeze(0)
        return out
