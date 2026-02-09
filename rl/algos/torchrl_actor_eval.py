from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import torch


class ActorEvalPolicy:
    def __init__(
        self,
        actor_backbone: torch.nn.Module,
        actor_head: torch.nn.Module,
        obs_scaler: torch.nn.Module,
        *,
        device: torch.device,
    ):
        self._actor_backbone = actor_backbone
        self._actor_head = actor_head
        self._obs_scaler = obs_scaler
        self._device = device

    def __call__(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self._device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            features = self._actor_backbone(self._obs_scaler(state_tensor))
            action_loc = self._actor_head(features)
            action = torch.tanh(action_loc)
        return action.squeeze(0).detach().cpu().numpy()


def capture_actor_snapshot(modules: Any) -> dict:
    backbone_state = {name: tensor.detach().clone() for name, tensor in modules.actor_backbone.state_dict().items()}
    head_state = {name: tensor.detach().clone() for name, tensor in modules.actor_head.state_dict().items()}
    return {
        "backbone": backbone_state,
        "head": head_state,
        "log_std": modules.log_std.detach().cpu().clone().numpy(),
    }


def restore_actor_snapshot(modules: Any, snapshot: dict, *, device: torch.device) -> None:
    modules.actor_backbone.load_state_dict(snapshot["backbone"])
    modules.actor_head.load_state_dict(snapshot["head"])
    modules.log_std.data.copy_(torch.as_tensor(snapshot["log_std"], device=device))


@contextmanager
def use_actor_snapshot(modules: Any, snapshot: dict, *, device: torch.device):
    previous_snapshot = capture_actor_snapshot(modules)
    restore_actor_snapshot(modules, snapshot, device=device)
    try:
        yield
    finally:
        restore_actor_snapshot(modules, previous_snapshot, device=device)
