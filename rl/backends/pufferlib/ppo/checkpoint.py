from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim


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
    payload: dict[str, Any] = {
        "iteration": int(iteration),
        "global_step": int(state.global_step),
        "actor_backbone": model.actor_backbone.state_dict(),
        "actor_head": model.actor_head.state_dict(),
        "critic_backbone": model.critic_backbone.state_dict(),
        "critic_head": model.critic_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_actor_state": state.best_actor_state,
        "best_return": float(state.best_return),
        "last_eval_return": float(state.last_eval_return),
        "last_heldout_return": _as_optional_finite(state.last_heldout_return),
        "last_episode_return": _as_optional_finite(state.last_episode_return),
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if hasattr(model, "log_std"):
        payload["log_std"] = model.log_std.detach().cpu()
    return payload


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
    checkpointing = __import__("rl.checkpointing", fromlist=["load_checkpoint"])
    loaded = checkpointing.load_checkpoint(Path(config.resume_from), device=device)
    model.actor_backbone.load_state_dict(loaded["actor_backbone"])
    model.actor_head.load_state_dict(loaded["actor_head"])
    model.critic_backbone.load_state_dict(loaded["critic_backbone"])
    model.critic_head.load_state_dict(loaded["critic_head"])
    if hasattr(model, "log_std") and "log_std" in loaded:
        model.log_std.data.copy_(torch.as_tensor(loaded["log_std"], device=device))
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
