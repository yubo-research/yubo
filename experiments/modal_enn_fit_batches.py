"""Fit-timing job keys, JSON payloads, and pending-job iteration for ENN Modal batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from analysis.fitting_time.fitting_time_enn_fit import EnnFitTimingResult
from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver
from experiments.enn_batch_job_params import (
    enn_batch_rep_meta_matches,
    enn_batch_shared_params,
    normalize_index_driver,
)

_FIT_META_REQUIRED: tuple[str, ...] = (
    "D",
    "function_name",
    "problem_seed",
    "data_seed",
    "rep_index",
    "num_reps",
    "index_driver",
)


def fit_job_key(
    *,
    d: int,
    function_name: str,
    n: int,
    problem_seed: int,
    rep_index: int,
    num_reps: int,
    index_driver: str | EnnIncrementalIndexDriver,
    normalize_function_name: Callable[[str], str],
) -> str:
    drv = normalize_index_driver(index_driver).value
    fn = normalize_function_name(function_name)
    return f"enn_fit_D{int(d)}_{fn}_N{int(n)}_pseed{int(problem_seed)}_nrep{int(num_reps)}_rep{int(rep_index)}_{drv}"


def fit_result_json_dest(
    output_dir: str | Path,
    *,
    d: int,
    function_name: str,
    n: int,
    problem_seed: int,
    rep_index: int,
    num_reps: int,
    index_driver: str | EnnIncrementalIndexDriver,
    normalize_function_name: Callable[[str], str],
) -> Path:
    return Path(output_dir) / (
        f"{fit_job_key(d=d, function_name=function_name, n=int(n), problem_seed=problem_seed, rep_index=rep_index, num_reps=num_reps, index_driver=index_driver, normalize_function_name=normalize_function_name)}.json"
    )


def fit_result_to_payload(
    result: EnnFitTimingResult,
    *,
    problem_seed: int,
    data_seed: int,
    rep_index: int,
    num_reps: int,
) -> dict:
    return {
        "N": int(result.n),
        "fit_seconds": float(result.fit_seconds),
        "log_likelihood": float(result.log_likelihood),
        "_meta": {
            "D": int(result.d),
            "function_name": result.target,
            "problem_seed": int(problem_seed),
            "data_seed": int(data_seed),
            "rep_index": int(rep_index),
            "num_reps": int(num_reps),
            "index_driver": result.index_driver.value,
        },
    }


def _fit_meta_matches(
    meta: dict,
    *,
    d: int,
    function_name: str,
    problem_seed: int,
    rep_index: int,
    num_reps: int,
    index_driver: str | EnnIncrementalIndexDriver,
    normalize_function_name: Callable[[str], str],
) -> bool:
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


def fit_result_json_complete(
    dest: str | Path,
    expected_n: int,
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
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if int(payload["N"]) != int(expected_n):
        return False
    try:
        float(payload["fit_seconds"])
        float(payload["log_likelihood"])
    except (KeyError, TypeError, ValueError):
        return False
    if not isinstance(meta, dict):
        return False
    if any(key not in meta for key in _FIT_META_REQUIRED):
        return False
    return _fit_meta_matches(
        meta,
        d=d,
        function_name=function_name,
        problem_seed=problem_seed,
        rep_index=rep_index,
        num_reps=num_reps,
        index_driver=index_driver,
        normalize_function_name=normalize_function_name,
    )


def iter_fit_jobs(
    output_dir: str | Path,
    index_driver: str,
    num_reps: int,
    d: int,
    problem_seed: int,
    *,
    iter_index_drivers: Callable[[str], tuple[EnnIncrementalIndexDriver, ...]],
    normalize_function_name: Callable[[str], str],
) -> Iterable[tuple[str, tuple[int, str, int, int, int, int, str]]]:
    shared = enn_batch_shared_params(num_reps=num_reps, d=d, problem_seed=problem_seed)
    d_i, nr, ps_i = shared.d, shared.num_reps, shared.problem_seed
    drvs = iter_index_drivers(index_driver)
    for fm in map(normalize_function_name, shared.benchmark_functions):
        for n in shared.checkpoint_ns:
            ni = int(n)
            for drv in drvs:
                for ri in range(nr):
                    dest = fit_result_json_dest(
                        output_dir,
                        d=d_i,
                        function_name=fm,
                        n=ni,
                        problem_seed=ps_i,
                        rep_index=ri,
                        num_reps=nr,
                        index_driver=drv,
                        normalize_function_name=normalize_function_name,
                    )
                    if fit_result_json_complete(
                        dest,
                        ni,
                        d=d_i,
                        function_name=fm,
                        problem_seed=ps_i,
                        rep_index=ri,
                        num_reps=nr,
                        index_driver=drv,
                        normalize_function_name=normalize_function_name,
                    ):
                        continue
                    yield (
                        fit_job_key(
                            d=d_i,
                            function_name=fm,
                            n=ni,
                            problem_seed=ps_i,
                            rep_index=ri,
                            num_reps=nr,
                            index_driver=drv,
                            normalize_function_name=normalize_function_name,
                        ),
                        (d_i, fm, ni, ps_i, ri, nr, drv.value),
                    )
