"""Incremental add-timing JSON completeness for ENN Modal batches."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.fitting_time.evaluate_metrics import normalize_benchmark_function_name
from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver

_ADD_META_REQUIRED: tuple[str, ...] = (
    "D",
    "function_name",
    "problem_seed",
    "rep_index",
    "num_reps",
    "index_driver",
)


def _normalize_index_driver(
    index_driver: str | EnnIncrementalIndexDriver,
) -> EnnIncrementalIndexDriver:
    if isinstance(index_driver, EnnIncrementalIndexDriver):
        return index_driver
    raw = str(index_driver).strip().lower().replace("-", "_")
    return EnnIncrementalIndexDriver(raw)


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
    drv = _normalize_index_driver(index_driver).value
    fn = normalize_benchmark_function_name(function_name)
    checks = (
        int(meta["D"]) == int(d),
        normalize_benchmark_function_name(str(meta["function_name"])) == fn,
        int(meta["problem_seed"]) == int(problem_seed),
        int(meta["rep_index"]) == int(rep_index),
        int(meta["num_reps"]) == int(num_reps),
        str(meta["index_driver"]).strip().lower() == drv,
    )
    return all(checks)


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
