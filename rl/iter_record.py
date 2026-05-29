from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def timing_record(*, iteration: int, frames_per_iter: int, elapsed: float, iter_dt: float) -> dict[str, Any]:
    step = int(iteration) * int(frames_per_iter)
    fps = float(frames_per_iter) / float(iter_dt) if iter_dt > 0 else float("nan")
    return {
        "iter": int(iteration),
        "step": step,
        "elapsed": float(elapsed),
        "iter_dt": float(iter_dt),
        "fps": fps,
    }


def merge_metric_fields(record: dict[str, Any], metrics: dict[str, float | None]) -> None:
    for key, value in metrics.items():
        if value is None:
            continue
        if key in {"rew", "ep_len", "ret_heldout"}:
            if np.isfinite(float(value)):
                record[key] = float(value)
            continue
        clean = finite_or_none(float(value))
        if clean is not None:
            record[key] = clean


@dataclass(frozen=True)
class IterInputs:
    iteration: int
    step: int
    frames_per_iter: int
    elapsed: float
    iter_dt: float
    metrics: dict[str, float | None]


@dataclass(frozen=True)
class EvalRecordInputs:
    started_at: float
    timing: dict[str, float | None]
    metrics: dict[str, float | None]
