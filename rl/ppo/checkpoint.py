from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.optim as optim

from rl.checkpointing import load_checkpoint, save_final_if_enabled, save_if_due
from rl.core import actor_state
from rl.ppo.metrics import _as_optional_finite


def build_checkpoint_payload(model, optimizer: optim.Optimizer, state, *, iteration: int) -> dict[str, Any]:
    actor_snapshot = actor_state.ppo_snap(
        model.actor_backbone,
        model.actor_head,
        log_std=model.log_std if hasattr(model, "log_std") else None,
    )
    return actor_state.build_ppo_checkpoint_payload(
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
        extra_payload={"last_episode_return": _as_optional_finite(state.last_episode_return)},
    )


def restore_checkpoint_if_requested(config, plan, model, optimizer: optim.Optimizer, state, *, device: torch.device) -> None:
    if not config.resume_from:
        return
    loaded = load_checkpoint(Path(config.resume_from), device=device)
    actor_snapshot: dict[str, Any] = {
        "backbone": loaded["actor_backbone"],
        "head": loaded["actor_head"],
    }
    if hasattr(model, "log_std") and "log_std" in loaded:
        actor_snapshot["log_std"] = loaded["log_std"]
    actor_state.load(
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
    actor_state.restore_rng_state_payload(loaded)


def maybe_save_periodic_checkpoint(
    config,
    checkpoint_manager: Any,
    model,
    optimizer: optim.Optimizer,
    state,
    *,
    iteration: int,
) -> None:
    save_if_due(
        checkpoint_manager,
        build_checkpoint_payload(model, optimizer, state, iteration=iteration),
        iteration=iteration,
        interval=config.checkpoint_interval,
    )


def save_final_checkpoint(
    config,
    checkpoint_manager: Any,
    model,
    optimizer: optim.Optimizer,
    state,
    *,
    iteration: int,
) -> None:
    save_final_if_enabled(
        checkpoint_manager,
        build_checkpoint_payload(model, optimizer, state, iteration=iteration),
        iteration=iteration,
        interval=config.checkpoint_interval,
    )
