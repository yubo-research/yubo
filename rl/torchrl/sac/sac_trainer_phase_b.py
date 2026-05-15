"""SAC training phase B (facade)."""

from __future__ import annotations

from .sac_trainer_phase_b_impl import (
    build_sac_collector,
    flatten_batch_to_transitions,
    normalize_actions_for_replay,
    process_sac_batch,
    run_sac_eval_log_checkpoint,
    update_step,
)

__all__ = [
    "build_sac_collector",
    "flatten_batch_to_transitions",
    "normalize_actions_for_replay",
    "process_sac_batch",
    "run_sac_eval_log_checkpoint",
    "update_step",
]
