"""SAC eval helpers (facade over :mod:`rl.pufferlib.sac.eval_utils_impl`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from common.im import im


_IMPL = "rl.pufferlib.sac.eval_utils_impl"

if TYPE_CHECKING:
    # Static export surface for Ruff F822; runtime stays lazy via ``__getattr__``.
    SacEvalPolicy: Any
    TrainState: Any
    append_eval_metric: Any
    build_eval_plan: Any
    capture_actor_state: Any
    collect_denoised_trajectory: Any
    due_mark: Any
    evaluate_actor: Any
    evaluate_for_best: Any
    evaluate_heldout_if_enabled: Any
    log_if_due: Any
    maybe_eval: Any
    render_videos_if_enabled: Any
    rl_logger: Any
    use_actor_state: Any

__all__ = [
    "SacEvalPolicy",
    "TrainState",
    "append_eval_metric",
    "build_eval_plan",
    "capture_actor_state",
    "collect_denoised_trajectory",
    "due_mark",
    "evaluate_actor",
    "evaluate_for_best",
    "evaluate_heldout_if_enabled",
    "log_if_due",
    "maybe_eval",
    "render_videos_if_enabled",
    "rl_logger",
    "use_actor_state",
]


def __getattr__(name: str):
    return getattr(im(_IMPL), name)


def __dir__():
    return sorted(__all__)
