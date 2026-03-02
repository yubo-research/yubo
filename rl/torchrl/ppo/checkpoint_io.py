from __future__ import annotations

from typing import Any

from rl.core.actor_state import (
    build_ppo_checkpoint_payload as build_shared_payload,
)
from rl.core.actor_state import (
    capture_ppo_actor_snapshot as capture_actor_snapshot,
)


def build_checkpoint_payload(
    training_setup: Any,
    modules: Any,
    train_state: Any,
    *,
    iteration: int,
) -> dict[str, Any]:
    actor_snapshot = capture_actor_snapshot(
        modules.actor_backbone,
        modules.actor_head,
        log_std=modules.log_std,
    )
    return build_shared_payload(
        iteration=iteration,
        global_step=int(iteration * training_setup.frames_per_batch),
        actor_snapshot=actor_snapshot,
        critic_backbone=modules.critic_backbone.state_dict(),
        critic_head=modules.critic_head.state_dict(),
        optimizer=training_setup.optimizer.state_dict(),
        best_actor_state=train_state.best_actor_state,
        best_return=float(train_state.best_return),
        last_eval_return=float(train_state.last_eval_return),
        last_heldout_return=train_state.last_heldout_return,
        extra_payload={"obs_scaler": modules.obs_scaler.state_dict()},
    )


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
