"""Shared SAC metric/log formatting helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from rl.core.progress import steps_per_second


def build_eval_metric_record(
    *,
    step: int,
    eval_return: float,
    heldout_return: float | None,
    best_return: float,
    loss_actor: float,
    loss_critic: float,
    loss_alpha: float,
    total_updates: int,
    started_at: float,
    now: float,
) -> dict[str, Any]:
    elapsed = float(now - float(started_at))
    return {
        "step": int(step),
        "eval_return": float(eval_return),
        "heldout_return": heldout_return,
        "best_return": float(best_return),
        "loss_actor": float(loss_actor),
        "loss_critic": float(loss_critic),
        "loss_alpha": float(loss_alpha),
        "total_updates": int(total_updates),
        "time_seconds": elapsed,
        "steps_per_second": float(steps_per_second(int(step), float(started_at), now=float(now))),
    }


def normalize_returns_for_log(
    *,
    eval_return: float,
    heldout_return: float | None,
    best_return: float,
) -> tuple[float | None, float | None, float]:
    eval_out = float(eval_return) if np.isfinite(float(eval_return)) else None
    best_out = float(best_return) if np.isfinite(float(best_return)) else 0.0
    return eval_out, heldout_return, best_out


def build_log_eval_iteration_kwargs(
    *,
    step: int,
    frames_per_batch: int,
    started_at: float,
    now: float,
    eval_return: float,
    heldout_return: float | None,
    best_return: float,
    loss_actor: float,
    loss_critic: float,
    loss_alpha: float,
) -> dict[str, Any]:
    eval_out, heldout_out, best_out = normalize_returns_for_log(
        eval_return=float(eval_return),
        heldout_return=heldout_return,
        best_return=float(best_return),
    )
    return {
        "iteration": 0,
        "num_iterations": 0,
        "frames_per_batch": int(frames_per_batch),
        "eval_return": eval_out,
        "heldout_return": heldout_out,
        "best_return": best_out,
        "algo_metrics": {
            "actor": float(loss_actor),
            "critic": float(loss_critic),
            "alpha": float(loss_alpha),
        },
        "algo_name": "sac",
        "elapsed": float(now - float(started_at)),
        "step_override": int(step),
    }
