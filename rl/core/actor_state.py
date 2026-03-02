from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import torch


def _clone_tensor(tensor: torch.Tensor, *, to_cpu: bool) -> torch.Tensor:
    out = tensor.detach()
    if bool(to_cpu):
        out = out.cpu()
    return out.clone()


def capture_backbone_head_snapshot(
    actor_backbone: torch.nn.Module,
    actor_head: torch.nn.Module,
    *,
    log_std: torch.Tensor | None = None,
    state_to_cpu: bool = False,
    log_std_to_cpu: bool = True,
    log_std_format: str = "tensor",
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "backbone": {name: _clone_tensor(tensor, to_cpu=state_to_cpu) for name, tensor in actor_backbone.state_dict().items()},
        "head": {name: _clone_tensor(tensor, to_cpu=state_to_cpu) for name, tensor in actor_head.state_dict().items()},
    }
    if log_std is None:
        return snapshot
    log_std_tensor = _clone_tensor(log_std, to_cpu=bool(log_std_to_cpu))
    fmt = str(log_std_format).strip().lower()
    if fmt == "tensor":
        snapshot["log_std"] = log_std_tensor
        return snapshot
    if fmt == "numpy":
        snapshot["log_std"] = log_std_tensor.numpy()
        return snapshot
    raise ValueError("log_std_format must be one of: tensor, numpy.")


def capture_ppo_actor_snapshot(actor_backbone: torch.nn.Module, actor_head: torch.nn.Module, *, log_std: torch.Tensor | None = None) -> dict[str, Any]:
    return capture_backbone_head_snapshot(actor_backbone, actor_head, log_std=log_std, state_to_cpu=False, log_std_to_cpu=True, log_std_format="tensor")


def restore_backbone_head_snapshot(
    actor_backbone: torch.nn.Module,
    actor_head: torch.nn.Module,
    snapshot: dict[str, Any],
    *,
    log_std: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> None:
    actor_backbone.load_state_dict(snapshot["backbone"])
    actor_head.load_state_dict(snapshot["head"])
    if log_std is not None and "log_std" in snapshot:
        target_device = log_std.device if device is None else device
        log_std.data.copy_(torch.as_tensor(snapshot["log_std"], device=target_device, dtype=log_std.dtype))


@contextmanager
def use_backbone_head_snapshot(
    actor_backbone: torch.nn.Module,
    actor_head: torch.nn.Module,
    snapshot: dict[str, Any],
    *,
    log_std: torch.Tensor | None = None,
    device: torch.device | None = None,
    state_to_cpu: bool = False,
    log_std_to_cpu: bool = True,
    log_std_format: str = "tensor",
):
    previous = capture_backbone_head_snapshot(
        actor_backbone, actor_head, log_std=log_std, state_to_cpu=state_to_cpu, log_std_to_cpu=log_std_to_cpu, log_std_format=log_std_format
    )
    restore_backbone_head_snapshot(actor_backbone, actor_head, snapshot, log_std=log_std, device=device)
    try:
        yield
    finally:
        restore_backbone_head_snapshot(actor_backbone, actor_head, previous, log_std=log_std, device=device)


def rng_state_payload() -> dict[str, Any]:
    return {
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def build_ppo_checkpoint_payload(
    *,
    iteration: int,
    global_step: int,
    actor_snapshot: dict[str, Any],
    critic_backbone: dict[str, Any],
    critic_head: dict[str, Any],
    optimizer: dict[str, Any],
    best_actor_state: dict[str, Any] | None,
    best_return: float,
    last_eval_return: float,
    last_heldout_return: float | None,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "iteration": int(iteration),
        "global_step": int(global_step),
        "actor_backbone": actor_snapshot["backbone"],
        "actor_head": actor_snapshot["head"],
        "critic_backbone": critic_backbone,
        "critic_head": critic_head,
        "optimizer": optimizer,
        "best_actor_state": best_actor_state,
        "best_return": float(best_return),
        "last_eval_return": float(last_eval_return),
        "last_heldout_return": last_heldout_return,
    }
    if "log_std" in actor_snapshot:
        payload["log_std"] = actor_snapshot["log_std"]
    payload.update(rng_state_payload())
    if extra_payload:
        payload.update(extra_payload)
    return payload
