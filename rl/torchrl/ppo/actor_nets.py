from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from rl.core.env_contract import ObservationContract

from .ppo_nets_base import (
    _BackboneHeadNet,
    _forward_backbone_features,
    _reshape_head_output,
)


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
        return (loc, scale)


class DiscreteActorNet(_BackboneHeadNet):
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
