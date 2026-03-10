from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from rl.core.actor_state import capture_backbone_head_snapshot, restore_backbone_head_snapshot, use_backbone_head_snapshot
from rl.core.pixel_transform import ensure_pixel_obs_format


class OffPolicyActorEvalPolicy:
    def __init__(
        self,
        actor_backbone: torch.nn.Module,
        actor_head: torch.nn.Module,
        obs_scaler: torch.nn.Module,
        *,
        act_dim: int,
        device: torch.device,
        from_pixels: bool = False,
        image_size: int = 84,
        channels: int = 3,
    ):
        self._actor_backbone = actor_backbone
        self._actor_head = actor_head
        self._obs_scaler = obs_scaler
        self._act_dim = int(act_dim)
        self._device = device
        self._from_pixels = bool(from_pixels)
        self._image_size = int(image_size)
        self._channels = int(channels)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.as_tensor(state, device=self._device)
        if self._from_pixels:
            state_tensor = ensure_pixel_obs_format(state_tensor, channels=self._channels, size=self._image_size, scale_float_255=True)
            if state_tensor.ndim == 3:
                state_tensor = state_tensor.unsqueeze(0)
        else:
            state_tensor = state_tensor.float()
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            features = self._actor_backbone(self._obs_scaler(state_tensor))
            head_output = self._actor_head(features)
            action_loc = head_output[..., : self._act_dim]
            action = torch.tanh(action_loc)
        return action.squeeze(0).detach().cpu().numpy()


def capture_actor_snapshot(modules: Any) -> dict:
    return capture_backbone_head_snapshot(modules.actor_backbone, modules.actor_head, log_std=None, state_to_cpu=False)


def restore_actor_snapshot(modules: Any, snapshot: dict) -> None:
    restore_backbone_head_snapshot(modules.actor_backbone, modules.actor_head, snapshot, log_std=None, device=None)


@contextmanager
def use_actor_snapshot(modules: Any, snapshot: dict, *, device: Any):
    with use_backbone_head_snapshot(modules.actor_backbone, modules.actor_head, snapshot, log_std=None, device=None, state_to_cpu=False):
        yield


SacActorEvalPolicy = OffPolicyActorEvalPolicy
capture_sac_actor_snapshot = capture_actor_snapshot
restore_sac_actor_snapshot = restore_actor_snapshot
use_sac_actor_snapshot = use_actor_snapshot
