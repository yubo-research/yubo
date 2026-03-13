from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _clone_tensor(tensor: torch.Tensor, *, to_cpu: bool) -> torch.Tensor:
    out = tensor.detach()
    if bool(to_cpu):
        out = out.cpu()
    return out.clone()


def snap(
    actor_backbone: nn.Module,
    actor_head: nn.Module,
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


def ppo_snap(
    actor_backbone: nn.Module,
    actor_head: nn.Module,
    *,
    log_std: torch.Tensor | None = None,
) -> dict[str, Any]:
    return snap(
        actor_backbone,
        actor_head,
        log_std=log_std,
        state_to_cpu=False,
        log_std_to_cpu=True,
        log_std_format="tensor",
    )


def load(
    actor_backbone: nn.Module,
    actor_head: nn.Module,
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
def using(
    actor_backbone: nn.Module,
    actor_head: nn.Module,
    snapshot: dict[str, Any],
    *,
    log_std: torch.Tensor | None = None,
    device: torch.device | None = None,
    state_to_cpu: bool = False,
    log_std_to_cpu: bool = True,
    log_std_format: str = "tensor",
):
    prev = snap(
        actor_backbone,
        actor_head,
        log_std=log_std,
        state_to_cpu=state_to_cpu,
        log_std_to_cpu=log_std_to_cpu,
        log_std_format=log_std_format,
    )
    load(actor_backbone, actor_head, snapshot, log_std=log_std, device=device)
    try:
        yield
    finally:
        load(actor_backbone, actor_head, prev, log_std=log_std, device=device)


def rng_state_payload() -> dict[str, Any]:
    rng_torch = torch.get_rng_state().detach().cpu().to(dtype=torch.uint8).contiguous()
    rng_cuda = None
    if torch.cuda.is_available():
        rng_cuda = [state.detach().cpu().to(dtype=torch.uint8).contiguous() for state in torch.cuda.get_rng_state_all()]
    return {
        "rng_torch": rng_torch,
        "rng_numpy": np.random.get_state(),
        "rng_cuda": rng_cuda,
    }


def _validate_rng_tensor(raw: Any, *, key: str) -> torch.Tensor:
    if not isinstance(raw, torch.Tensor):
        raise TypeError(f"Checkpoint field '{key}' must be a torch.Tensor, got {type(raw).__name__}.")
    if raw.dtype != torch.uint8:
        raise TypeError(f"Checkpoint field '{key}' must have dtype torch.uint8, got {raw.dtype}.")
    if raw.ndim != 1:
        raise ValueError(f"Checkpoint field '{key}' must be 1D, got shape {tuple(raw.shape)}.")
    return raw.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def restore_rng_state_payload(payload: dict[str, Any]) -> None:
    if "rng_torch" in payload:
        torch.set_rng_state(_validate_rng_tensor(payload["rng_torch"], key="rng_torch"))
    if "rng_numpy" in payload:
        np.random.set_state(payload["rng_numpy"])
    if torch.cuda.is_available() and payload.get("rng_cuda") is not None:
        rng_cuda = payload["rng_cuda"]
        if not isinstance(rng_cuda, (list, tuple)):
            raise TypeError(f"Checkpoint field 'rng_cuda' must be a list/tuple of tensors, got {type(rng_cuda).__name__}.")
        states = [_validate_rng_tensor(state, key=f"rng_cuda[{idx}]") for idx, state in enumerate(rng_cuda)]
        torch.cuda.set_rng_state_all(states)


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
