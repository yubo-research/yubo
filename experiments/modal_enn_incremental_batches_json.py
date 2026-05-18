"""Incremental add-timing JSON completeness for ENN Modal batches."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.fitting_time.evaluate_metrics import normalize_benchmark_function_name
from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver
from experiments.enn_batch_job_params import enn_batch_rep_meta_matches

_ADD_META_REQUIRED: tuple[str, ...] = (
    "D",
    "function_name",
    "problem_seed",
    "data_seed",
    "rep_index",
    "num_reps",
    "index_driver",
)


def add_meta_matches(
    meta: dict,
    *,
    d: int,
    function_name: str,
    problem_seed: int,
    rep_index: int,
    num_reps: int,
    index_driver: str | EnnIncrementalIndexDriver,
) -> bool:
    return enn_batch_rep_meta_matches(
        meta,
        d=d,
        function_name=function_name,
        problem_seed=problem_seed,
        rep_index=rep_index,
        num_reps=num_reps,
        index_driver=index_driver,
        normalize_function_name=normalize_benchmark_function_name,
    )


def result_json_complete(
    dest: str | Path,
    expected_n: tuple[int, ...],
    *,
    d: int,
    function_name: str,
    problem_seed: int,
    rep_index: int,
    num_reps: int,
    index_driver: str | EnnIncrementalIndexDriver,
) -> bool:
    path = Path(dest)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
        meta = payload["_meta"]
        n_vals = tuple(int(n) for n in payload["N"])
        av, lv = payload["add_seconds"], payload["log_likelihood"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    tn = tuple(int(n) for n in expected_n)
    if n_vals != tn or len(av) != len(tn) or len(lv) != len(tn):
        return False
    if not isinstance(meta, dict):
        return False
    if any(key not in meta for key in _ADD_META_REQUIRED):
        return False
    return add_meta_matches(
        meta,
        d=d,
        function_name=function_name,
        problem_seed=problem_seed,
        rep_index=rep_index,
        num_reps=num_reps,
        index_driver=index_driver,
    )
