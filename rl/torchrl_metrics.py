from __future__ import annotations

from typing import Any

from rl.iter_record import IterInputs, merge_metric_fields, timing_record


def build_ppo_iter_record(inputs: IterInputs) -> dict[str, Any]:
    """Build one TorchRL PPO iteration record (jsonl + ITER: console line)."""
    record = timing_record(
        iteration=inputs.iteration,
        frames_per_iter=inputs.frames_per_iter,
        elapsed=inputs.elapsed,
        iter_dt=inputs.iter_dt,
    )
    record["step"] = int(inputs.step)
    merge_metric_fields(record, inputs.metrics)
    return record


def build_sac_iter_record(inputs: IterInputs) -> dict[str, Any]:
    """Build one TorchRL SAC step record (jsonl + ITER: console line)."""
    frames = max(1, int(inputs.frames_per_iter))
    record = timing_record(
        iteration=int(inputs.iteration),
        frames_per_iter=frames,
        elapsed=inputs.elapsed,
        iter_dt=inputs.iter_dt,
    )
    record["step"] = int(inputs.step)
    merge_metric_fields(record, inputs.metrics)
    return record
