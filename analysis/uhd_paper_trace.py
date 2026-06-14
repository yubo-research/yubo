"""Trace-list utilities for UHD paper notebooks (JSONL ``TraceRecord`` reps)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from analysis.data_io import TraceRecord


def sem_1d(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def traces_to_array(traces: Sequence[Sequence[TraceRecord]]) -> np.ndarray:
    """Per-rep eval returns at each iteration (stored as ``rreturn`` in JSONL)."""
    if not traces:
        return np.empty((0, 0))
    max_len = max(len(t) for t in traces)
    arr = np.full((len(traces), max_len), np.nan)
    for i, trace in enumerate(traces):
        for j, record in enumerate(trace):
            arr[i, j] = record.rreturn
    return arr


def best_so_far_array(traces: Sequence[Sequence[TraceRecord]]) -> np.ndarray:
    """Cumulative max eval return within each rep."""
    arr = traces_to_array(traces)
    if arr.size == 0:
        return arr
    return np.maximum.accumulate(arr, axis=1)


def summarize_curve(
    traces: Sequence[Sequence[TraceRecord]],
) -> dict[str, np.ndarray | int]:
    arr = best_so_far_array(traces)
    n_per_iter = np.sum(np.isfinite(arr), axis=0)
    sem = np.full(arr.shape[1], np.nan)
    ok = n_per_iter > 1
    if arr.shape[1] and np.any(ok):
        sem[ok] = np.nanstd(arr[:, ok], axis=0, ddof=1) / np.sqrt(n_per_iter[ok])
    return {
        "n": arr.shape[0],
        "iterations": np.arange(arr.shape[1]),
        "mean": np.nanmean(arr, axis=0),
        "sem": sem,
        "q25": np.nanpercentile(arr, 25, axis=0),
        "q75": np.nanpercentile(arr, 75, axis=0),
    }
