from __future__ import annotations

import time


def is_due(step: int, interval: int | None) -> bool:
    return interval is not None and int(interval) > 0 and (int(step) % int(interval) == 0)


def due_mark(step: int, interval: int | None, previous_mark: int) -> int | None:
    if interval is None or int(interval) <= 0:
        return None
    interval_i = int(interval)
    current_mark = int(step // interval_i)
    if current_mark <= 0 or current_mark <= int(previous_mark):
        return None
    return current_mark


def steps_per_second(step: int, started_at: float, *, now: float | None = None) -> float:
    t_now = float(time.time()) if now is None else float(now)
    elapsed = float(t_now - float(started_at))
    if elapsed <= 0.0:
        return float("nan")
    return float(step) / elapsed
