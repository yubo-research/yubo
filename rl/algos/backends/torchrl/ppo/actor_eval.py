from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from ..common.env_contract import ObservationContract
from ..common.pixel_transform import ensure_pixel_obs_format


class ActorEvalPolicy:
    def __init__(
        self,
        actor_backbone: torch.nn.Module,
        actor_head: torch.nn.Module,
        obs_scaler: torch.nn.Module,
        *,
        device: torch.device,
        obs_contract: ObservationContract,
        is_discrete: bool = False,
    ):
        self._actor_backbone = actor_backbone
        self._actor_head = actor_head
        self._obs_scaler = obs_scaler
        self._device = device
        self._obs_contract = obs_contract
        self._is_discrete = bool(is_discrete)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.as_tensor(state, device=self._device)
        if self._obs_contract.mode == "pixels":
            state_tensor = ensure_pixel_obs_format(
                state_tensor,
                channels=int(self._obs_contract.model_channels or 3),
                size=int(self._obs_contract.image_size or 84),
                scale_float_255=True,
            )
            if state_tensor.ndim == 3:
                state_tensor = state_tensor.unsqueeze(0)
        else:
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
            state_tensor = state_tensor.float()
        with torch.no_grad():
            features = self._actor_backbone(self._obs_scaler(state_tensor))
            head_out = self._actor_head(features)
            if self._is_discrete:
                action = head_out.argmax(dim=-1).float()
            else:
                action = torch.tanh(head_out)
        return action.squeeze(0).detach().cpu().numpy()


def capture_actor_snapshot(modules: Any) -> dict:
    backbone_state = {name: tensor.detach().clone() for name, tensor in modules.actor_backbone.state_dict().items()}
    head_state = {name: tensor.detach().clone() for name, tensor in modules.actor_head.state_dict().items()}
    snapshot = {"backbone": backbone_state, "head": head_state}
    if modules.log_std is not None:
        snapshot["log_std"] = modules.log_std.detach().cpu().clone().numpy()
    return snapshot


def restore_actor_snapshot(modules: Any, snapshot: dict, *, device: torch.device) -> None:
    modules.actor_backbone.load_state_dict(snapshot["backbone"])
    modules.actor_head.load_state_dict(snapshot["head"])
    if modules.log_std is not None and "log_std" in snapshot:
        modules.log_std.data.copy_(torch.as_tensor(snapshot["log_std"], device=device))


@contextmanager
def use_actor_snapshot(modules: Any, snapshot: dict, *, device: torch.device):
    previous_snapshot = capture_actor_snapshot(modules)
    restore_actor_snapshot(modules, snapshot, device=device)
    try:
        yield
    finally:
        restore_actor_snapshot(modules, previous_snapshot, device=device)
