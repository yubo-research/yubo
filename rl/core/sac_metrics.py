from __future__ import annotations

from typing import Any

import numpy as np

from rl.iter_record import EvalRecordInputs


def build_eval_metric_record(ctx: EvalRecordInputs) -> dict[str, Any]:
    from rl.iter_record import IterInputs
    from rl.torchrl_metrics import build_sac_iter_record

    step = int(ctx.timing["step"])
    elapsed = float(ctx.timing["now"]) - float(ctx.started_at)
    frames = int(ctx.timing.get("frames_per_iter") or 1)
    dt = float(ctx.timing.get("iter_dt") or max(elapsed / max(1, step), 1e-9))
    return build_sac_iter_record(
        IterInputs(
            iteration=int(step) // max(1, frames),
            step=step,
            frames_per_iter=frames,
            elapsed=elapsed,
            iter_dt=dt,
            metrics=dict(ctx.metrics),
        )
    )


def normalize_returns_for_log(*, eval_return: float, heldout_return: float | None, best_return: float) -> tuple[float | None, float | None, float]:
    eval_out = float(eval_return) if np.isfinite(float(eval_return)) else None
    best_out = float(best_return) if np.isfinite(float(best_return)) else 0.0
    return (eval_out, heldout_return, best_out)


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
