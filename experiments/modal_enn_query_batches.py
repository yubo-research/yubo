"""Query-timing job keys, JSON payloads, and pending-job iteration for ENN Modal batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver
from analysis.fitting_time.fitting_time_enn_query import EnnQueryTimingResult
from experiments.enn_batch_job_params import (
    enn_batch_rep_meta_matches,
    normalize_index_driver,
)
from experiments.modal_enn_series_batches import iter_replicate_series_jobs

_QUERY_META_REQUIRED: tuple[str, ...] = (
    "D",
    "function_name",
    "problem_seed",
    "data_seed",
    "rep_index",
    "num_reps",
    "index_driver",
    "num_query_points",
)


def query_job_key(
    *,
    d: int,
    function_name: str,
    problem_seed: int,
    rep_index: int,
    num_reps: int,
    index_driver: str | EnnIncrementalIndexDriver,
    normalize_function_name: Callable[[str], str],
) -> str:
    drv = normalize_index_driver(index_driver).value
    fn = normalize_function_name(function_name)
    return f"enn_query_D{int(d)}_{fn}_pseed{int(problem_seed)}_nrep{int(num_reps)}_rep{int(rep_index)}_{drv}"


def query_result_json_dest(
    output_dir: str | Path,
    *,
    d: int,
    function_name: str,
    problem_seed: int,
    rep_index: int,
    num_reps: int,
    index_driver: str | EnnIncrementalIndexDriver,
    normalize_function_name: Callable[[str], str],
) -> Path:
    return Path(output_dir) / (
        f"{query_job_key(d=d, function_name=function_name, problem_seed=problem_seed, rep_index=rep_index, num_reps=num_reps, index_driver=index_driver, normalize_function_name=normalize_function_name)}.json"
    )


def query_result_to_payload(
    result: EnnQueryTimingResult,
    *,
    problem_seed: int,
    data_seed: int,
    rep_index: int,
    num_reps: int,
) -> dict:
    return {
        "N": list(result.n),
        "query_seconds": list(result.query_seconds),
        "query_seconds_per_point": list(result.query_seconds_per_point),
        "_meta": {
            "D": int(result.d),
            "function_name": result.target,
            "problem_seed": int(problem_seed),
            "data_seed": int(data_seed),
            "rep_index": int(rep_index),
            "num_reps": int(num_reps),
            "index_driver": result.index_driver.value,
            "num_query_points": int(result.num_query_points),
        },
    }


def query_result_json_complete(
    dest: str | Path,
    expected_n: tuple[int, ...],
    *,
    d: int,
    function_name: str,
    problem_seed: int,
    rep_index: int,
    num_reps: int,
    index_driver: str | EnnIncrementalIndexDriver,
    normalize_function_name: Callable[[str], str],
) -> bool:
    path = Path(dest)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
        meta = payload["_meta"]
        ns = tuple(int(n) for n in payload["N"])
        query_seconds = tuple(float(v) for v in payload["query_seconds"])
        per_point = tuple(float(v) for v in payload["query_seconds_per_point"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if ns != tuple(int(n) for n in expected_n):
        return False
    if len(query_seconds) != len(ns) or len(per_point) != len(ns):
        return False
    if not isinstance(meta, dict):
        return False
    if any(key not in meta for key in _QUERY_META_REQUIRED):
        return False
    return enn_batch_rep_meta_matches(
        meta,
        d=d,
        function_name=function_name,
        problem_seed=problem_seed,
        rep_index=rep_index,
        num_reps=num_reps,
        index_driver=index_driver,
        normalize_function_name=normalize_function_name,
    )


def iter_query_jobs(
    output_dir: str | Path,
    index_driver: str,
    num_reps: int,
    d: int,
    problem_seed: int,
    *,
    iter_index_drivers: Callable[[str], tuple[EnnIncrementalIndexDriver, ...]],
    normalize_function_name: Callable[[str], str],
) -> Iterable[tuple[str, tuple[int, str, int, int, int, str]]]:
    def complete(dest, expected_n, **kwargs):
        return query_result_json_complete(
            dest,
            expected_n,
            normalize_function_name=normalize_function_name,
            **kwargs,
        )

    yield from iter_replicate_series_jobs(
        output_dir,
        index_driver,
        num_reps,
        d,
        problem_seed,
        iter_index_drivers=iter_index_drivers,
        normalize_function_name=normalize_function_name,
        result_json_dest=query_result_json_dest,
        result_json_complete=complete,
        job_key=query_job_key,
    )
