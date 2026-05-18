"""Fit-timing job keys, JSON payloads, and pending-job iteration for ENN Modal batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from analysis.fitting_time.fitting_time_enn_fit import EnnFitTimingResult, enn_fit_quality_ns
from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver

_FIT_META_REQUIRED: tuple[str, ...] = (
    "D",
    "function_name",
    "problem_seed",
    "data_seed",
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
    try:
        return EnnIncrementalIndexDriver(raw)
    except ValueError as exc:
        valid = ", ".join(d.value for d in EnnIncrementalIndexDriver)
        raise ValueError(f"unknown index_driver={index_driver!r}; expected one of {valid}") from exc


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
    drv = _normalize_index_driver(index_driver).value
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
    drv = _normalize_index_driver(index_driver).value
    fn = normalize_function_name(function_name)
    checks = (
        int(meta["D"]) == int(d),
        normalize_function_name(str(meta["function_name"])) == fn,
        int(meta["problem_seed"]) == int(problem_seed),
        int(meta["rep_index"]) == int(rep_index),
        int(meta["num_reps"]) == int(num_reps),
        str(meta["index_driver"]).strip().lower() == drv,
    )
    return all(checks)


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
    except (TypeError, ValueError):
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
    benchmark_functions: tuple[str, ...],
    iter_index_drivers: Callable[[str], tuple[EnnIncrementalIndexDriver, ...]],
    normalize_function_name: Callable[[str], str],
) -> Iterable[tuple[str, tuple[int, str, int, int, int, int, str]]]:
    if int(num_reps) < 1:
        raise ValueError("num_reps must be >= 1")
    if int(d) < 1:
        raise ValueError("D must be positive")
    drvs, d_i = iter_index_drivers(index_driver), int(d)
    ps_i, nr = int(problem_seed), int(num_reps)
    for fm in map(normalize_function_name, benchmark_functions):
        for n in enn_fit_quality_ns():
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
