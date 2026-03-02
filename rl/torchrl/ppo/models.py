from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from rl.core.env_contract import ObservationContract
from rl.core.pixel_transform import ensure_pixel_obs_format


def _flatten_pixel_batch(
    obs: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...] | None]:
    batch_shape: tuple[int, ...] | None = None
    if obs.ndim == 5:
        batch_shape = obs.shape[:2]
        obs = obs.flatten(0, 1)
    return obs, batch_shape


def _forward_backbone_features(
    obs: torch.Tensor,
    *,
    obs_scaler: Any,
    obs_contract: ObservationContract,
    backbone: nn.Module,
) -> tuple[torch.Tensor, tuple[int, ...] | None, bool]:
    obs = obs_scaler(obs)
    obs, batch_shape, squeeze_batch_dim = prepare_obs_for_backbone(obs, obs_contract)
    feats = backbone(obs)
    return feats, batch_shape, squeeze_batch_dim


def _reshape_head_output(
    output: torch.Tensor,
    *,
    batch_shape: tuple[int, ...] | None,
    squeeze_batch_dim: bool,
) -> torch.Tensor:
    if batch_shape is not None:
        return output.reshape(*batch_shape, -1)
    if squeeze_batch_dim:
        return output.squeeze(0)
    return output


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
        feats, batch_shape, squeeze_batch_dim = _forward_backbone_features(
            obs,
            obs_scaler=self.obs_scaler,
            obs_contract=self.obs_contract,
            backbone=self.backbone,
        )
        loc = self.head(feats)
        scale = self.log_std.exp().expand_as(loc)
        loc = _reshape_head_output(loc, batch_shape=batch_shape, squeeze_batch_dim=squeeze_batch_dim)
        scale = _reshape_head_output(scale, batch_shape=batch_shape, squeeze_batch_dim=squeeze_batch_dim)
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
        feats, batch_shape, squeeze_batch_dim = _forward_backbone_features(
            obs,
            obs_scaler=self.obs_scaler,
            obs_contract=self.obs_contract,
            backbone=self.backbone,
        )
        logits = self.head(feats)
        logits = _reshape_head_output(logits, batch_shape=batch_shape, squeeze_batch_dim=squeeze_batch_dim)
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
        feats, batch_shape, squeeze_batch_dim = _forward_backbone_features(
            obs,
            obs_scaler=self.obs_scaler,
            obs_contract=self.obs_contract,
            backbone=self.backbone,
        )
        out = self.head(feats)
        out = _reshape_head_output(out, batch_shape=batch_shape, squeeze_batch_dim=squeeze_batch_dim)
        return out
