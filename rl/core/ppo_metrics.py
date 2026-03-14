from __future__ import annotations

import time
from typing import Any

from rl.core.progress import steps_per_second


def record(
    *,
    iteration: int,
    global_step: int,
    eval_return: float | None,
    heldout_return: float | None,
    best_return: float | None,
    approx_kl: float | None,
    clipfrac: float | None,
    started_at: float,
    now: float | None = None,
) -> dict[str, Any]:
    t_now = float(time.time()) if now is None else float(now)
    elapsed = float(t_now - float(started_at))
    return {
        "iteration": int(iteration),
        "global_step": int(global_step),
        "eval_return": eval_return,
        "heldout_return": heldout_return,
        "best_return": best_return,
        "approx_kl": approx_kl,
        "clipfrac": clipfrac,
        "time_seconds": elapsed,
        "steps_per_second": float(steps_per_second(int(global_step), float(started_at), now=t_now)),
    }
