from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim

from rl.checkpointing import load_checkpoint
from rl.core.actor_state import (
    build_ppo_checkpoint_payload as build_shared_payload,
)
from rl.core.actor_state import (
    capture_ppo_actor_snapshot as capture_actor_snapshot,
)
from rl.core.actor_state import (
    restore_backbone_head_snapshot,
)


def _as_optional_finite(value: float | None) -> float | None:
    if value is None:
        return None
    value_f = float(value)
    if not np.isfinite(value_f):
        return None
    return value_f


def build_checkpoint_payload(
    model,
    optimizer: optim.Optimizer,
    state,
    *,
    iteration: int,
) -> dict[str, Any]:
    actor_snapshot = capture_actor_snapshot(
        model.actor_backbone,
        model.actor_head,
        log_std=model.log_std if hasattr(model, "log_std") else None,
    )
    return build_shared_payload(
        iteration=iteration,
        global_step=int(state.global_step),
        actor_snapshot=actor_snapshot,
        critic_backbone=model.critic_backbone.state_dict(),
        critic_head=model.critic_head.state_dict(),
        optimizer=optimizer.state_dict(),
        best_actor_state=state.best_actor_state,
        best_return=float(state.best_return),
        last_eval_return=float(state.last_eval_return),
        last_heldout_return=_as_optional_finite(state.last_heldout_return),
        extra_payload={
            "last_episode_return": _as_optional_finite(state.last_episode_return),
        },
    )


def restore_checkpoint_if_requested(
    config,
    plan,
    model,
    optimizer: optim.Optimizer,
    state,
    *,
    device: torch.device,
) -> None:
    if not config.resume_from:
        return
    loaded = load_checkpoint(Path(config.resume_from), device=device)
    actor_snapshot: dict[str, Any] = {
        "backbone": loaded["actor_backbone"],
        "head": loaded["actor_head"],
    }
    if hasattr(model, "log_std") and "log_std" in loaded:
        actor_snapshot["log_std"] = loaded["log_std"]
    restore_backbone_head_snapshot(
        model.actor_backbone,
        model.actor_head,
        actor_snapshot,
        log_std=model.log_std if hasattr(model, "log_std") else None,
        device=device,
    )
    model.critic_backbone.load_state_dict(loaded["critic_backbone"])
    model.critic_head.load_state_dict(loaded["critic_head"])
    optimizer.load_state_dict(loaded["optimizer"])

    state.start_iteration = int(loaded.get("iteration", 0))
    state.global_step = int(loaded.get("global_step", state.start_iteration * plan.batch_size))
    state.best_actor_state = loaded.get("best_actor_state")
    if "best_return" in loaded:
        state.best_return = float(loaded["best_return"])
    if "last_eval_return" in loaded:
        state.last_eval_return = float(loaded["last_eval_return"])
    if "last_heldout_return" in loaded:
        state.last_heldout_return = _as_optional_finite(loaded["last_heldout_return"])
    if "last_episode_return" in loaded and loaded["last_episode_return"] is not None:
        state.last_episode_return = float(loaded["last_episode_return"])

    if "rng_torch" in loaded:
        torch.set_rng_state(loaded["rng_torch"])
    if "rng_numpy" in loaded:
        np.random.set_state(loaded["rng_numpy"])
    if torch.cuda.is_available() and loaded.get("rng_cuda") is not None:
        torch.cuda.set_rng_state_all(loaded["rng_cuda"])


def maybe_save_periodic_checkpoint(
    config,
    checkpoint_manager: Any,
    model,
    optimizer: optim.Optimizer,
    state,
    *,
    iteration: int,
) -> None:
    if config.checkpoint_interval is None:
        return
    interval = int(config.checkpoint_interval)
    if interval <= 0 or iteration % interval != 0:
        return
    payload = build_checkpoint_payload(model, optimizer, state, iteration=iteration)
    checkpoint_manager.save_both(payload, iteration=iteration)


def save_final_checkpoint(
    config,
    checkpoint_manager: Any,
    model,
    optimizer: optim.Optimizer,
    state,
    *,
    iteration: int,
) -> None:
    if config.checkpoint_interval is None:
        return
    payload = build_checkpoint_payload(model, optimizer, state, iteration=iteration)
    checkpoint_manager.save_both(payload, iteration=iteration)
