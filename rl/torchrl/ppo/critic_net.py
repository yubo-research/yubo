from __future__ import annotations

import torch

from .ppo_nets_base import _BackboneHeadNet, _forward_backbone_features, _reshape_head_output


class CriticNet(_BackboneHeadNet):
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
