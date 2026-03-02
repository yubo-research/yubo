from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from rl.core.actor_state import (
    capture_backbone_head_snapshot,
    restore_backbone_head_snapshot,
    use_backbone_head_snapshot,
)
from rl.core.env_contract import ObservationContract
from rl.core.pixel_transform import ensure_pixel_obs_format


def _has_state_dict(module: Any) -> bool:
    return hasattr(module, "state_dict") and callable(getattr(module, "state_dict"))


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
        if self._obs_contract.mode == "pixels":
            state_tensor = torch.as_tensor(state, device=self._device)
            state_tensor = ensure_pixel_obs_format(
                state_tensor,
                channels=int(self._obs_contract.model_channels or 3),
                size=int(self._obs_contract.image_size or 84),
                scale_float_255=True,
            )
            if state_tensor.ndim == 3:
                state_tensor = state_tensor.unsqueeze(0)
        else:
            # MPS doesn't support float64; force vector eval state to float32.
            state_tensor = torch.as_tensor(state, device=self._device, dtype=torch.float32)
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            features = self._actor_backbone(self._obs_scaler(state_tensor))
            head_out = self._actor_head(features)
            if self._is_discrete:
                action = head_out.argmax(dim=-1).float()
            else:
                action = torch.tanh(head_out)
        return action.squeeze(0).detach().cpu().numpy()


def capture_actor_snapshot(modules: Any) -> dict:
    if not _has_state_dict(modules.actor_backbone) or not _has_state_dict(modules.actor_head):
        return {}
    return capture_backbone_head_snapshot(
        modules.actor_backbone,
        modules.actor_head,
        log_std=getattr(modules, "log_std", None),
        state_to_cpu=False,
        log_std_to_cpu=True,
        log_std_format="numpy",
    )


def restore_actor_snapshot(modules: Any, snapshot: dict, *, device: torch.device) -> None:
    if not _has_state_dict(modules.actor_backbone) or not _has_state_dict(modules.actor_head):
        return
    restore_backbone_head_snapshot(
        modules.actor_backbone,
        modules.actor_head,
        snapshot,
        log_std=getattr(modules, "log_std", None),
        device=device,
    )


@contextmanager
def use_actor_snapshot(modules: Any, snapshot: dict, *, device: torch.device):
    if not _has_state_dict(modules.actor_backbone) or not _has_state_dict(modules.actor_head):
        yield
        return
    with use_backbone_head_snapshot(
        modules.actor_backbone,
        modules.actor_head,
        snapshot,
        log_std=getattr(modules, "log_std", None),
        device=device,
        state_to_cpu=False,
        log_std_to_cpu=True,
        log_std_format="numpy",
    ):
        yield
