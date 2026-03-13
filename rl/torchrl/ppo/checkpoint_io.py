from __future__ import annotations

from typing import Any

from rl.checkpointing import save_final_if_enabled, save_if_due
from rl.core.actor_state import build_ppo_checkpoint_payload as build_shared_payload
from rl.core.actor_state import ppo_snap as capture_actor_snapshot


def build_checkpoint_payload(training_setup: Any, modules: Any, train_state: Any, *, iteration: int) -> dict[str, Any]:
    actor_snapshot = capture_actor_snapshot(modules.actor_backbone, modules.actor_head, log_std=modules.log_std)
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


def save_periodic_checkpoint(config: Any, training_setup: Any, modules: Any, train_state: Any, *, iteration: int) -> None:
    save_if_due(
        training_setup.checkpoint_manager,
        build_checkpoint_payload(training_setup, modules, train_state, iteration=iteration),
        iteration=iteration,
        interval=config.checkpoint_interval,
    )


def save_final_checkpoint(config: Any, training_setup: Any, modules: Any, train_state: Any) -> None:
    save_final_if_enabled(
        training_setup.checkpoint_manager,
        build_checkpoint_payload(
            training_setup,
            modules,
            train_state,
            iteration=int(training_setup.num_iterations),
        ),
        iteration=int(training_setup.num_iterations),
        interval=config.checkpoint_interval,
    )
