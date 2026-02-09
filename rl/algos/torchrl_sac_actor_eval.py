from __future__ import annotations

from typing import Any

import numpy as np
import torch


class SacActorEvalPolicy:
    def __init__(
        self,
        actor_backbone: torch.nn.Module,
        actor_head: torch.nn.Module,
        obs_scaler: torch.nn.Module,
        *,
        act_dim: int,
        device: torch.device,
    ):
        self._actor_backbone = actor_backbone
        self._actor_head = actor_head
        self._obs_scaler = obs_scaler
        self._act_dim = int(act_dim)
        self._device = device

    def __call__(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self._device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            features = self._actor_backbone(self._obs_scaler(state_tensor))
            head_output = self._actor_head(features)
            action_loc = head_output[..., : self._act_dim]
            action = torch.tanh(action_loc)
        return action.squeeze(0).detach().cpu().numpy()


def capture_sac_actor_snapshot(modules: Any) -> dict:
    return {
        "backbone": {name: tensor.detach().clone() for name, tensor in modules.actor_backbone.state_dict().items()},
        "head": {name: tensor.detach().clone() for name, tensor in modules.actor_head.state_dict().items()},
    }


def restore_sac_actor_snapshot(modules: Any, snapshot: dict) -> None:
    modules.actor_backbone.load_state_dict(snapshot["backbone"])
    modules.actor_head.load_state_dict(snapshot["head"])
