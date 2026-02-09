from __future__ import annotations

from typing import Any

import numpy as np
import torch


def build_checkpoint_payload(
    training_setup: Any,
    modules: Any,
    train_state: Any,
    *,
    iteration: int,
) -> dict[str, Any]:
    return {
        "iteration": int(iteration),
        "global_step": int(iteration * training_setup.frames_per_batch),
        "actor_backbone": modules.actor_backbone.state_dict(),
        "actor_head": modules.actor_head.state_dict(),
        "critic_backbone": modules.critic_backbone.state_dict(),
        "critic_head": modules.critic_head.state_dict(),
        "log_std": modules.log_std.detach().cpu(),
        "optimizer": training_setup.optimizer.state_dict(),
        "obs_scaler": modules.obs_scaler.state_dict(),
        "best_return": float(train_state.best_return),
        "best_actor_state": train_state.best_actor_state,
        "last_eval_return": float(train_state.last_eval_return),
        "last_heldout_return": train_state.last_heldout_return,
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_periodic_checkpoint(
    config: Any,
    training_setup: Any,
    modules: Any,
    train_state: Any,
    *,
    iteration: int,
) -> None:
    if not config.checkpoint_interval or iteration % int(config.checkpoint_interval) != 0:
        return
    payload = build_checkpoint_payload(training_setup, modules, train_state, iteration=iteration)
    training_setup.checkpoint_manager.save_both(payload, iteration=iteration)


def save_final_checkpoint(
    config: Any,
    training_setup: Any,
    modules: Any,
    train_state: Any,
) -> None:
    if not config.checkpoint_interval:
        return
    payload = build_checkpoint_payload(
        training_setup,
        modules,
        train_state,
        iteration=int(training_setup.num_iterations),
    )
    training_setup.checkpoint_manager.save_both(payload, iteration=int(training_setup.num_iterations))
