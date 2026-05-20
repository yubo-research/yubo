"""Fit-ind job keys, JSON payloads, and pending-job iteration for ENN Modal batches."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from analysis.fitting_time.fitting_time_enn_fit_ind import EnnFitIndTimingResult
from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver
from experiments import modal_enn_fit_ind_batches_json as _fit_ind_json
from experiments.enn_batch_job_params import (
    enn_batch_shared_params,
    normalize_index_driver,
)


def fit_ind_job_key(
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
    return f"enn_fit_ind_D{int(d)}_{fn}_pseed{int(problem_seed)}_nrep{int(num_reps)}_rep{int(rep_index)}_{drv}"


def fit_ind_result_json_dest(
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
        f"{fit_ind_job_key(d=d, function_name=function_name, problem_seed=problem_seed, rep_index=rep_index, num_reps=num_reps, index_driver=index_driver, normalize_function_name=normalize_function_name)}.json"
    )


def fit_ind_result_to_payload(
    result: EnnFitIndTimingResult,
    *,
    problem_seed: int,
    data_seed: int,
    rep_index: int,
    num_reps: int,
) -> dict:
    return {
        "N": list(result.n),
        "fit_seconds": list(result.fit_seconds),
        "log_likelihood": list(result.log_likelihood),
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


def iter_fit_ind_jobs(
    output_dir: str | Path,
    index_driver: str,
    num_reps: int,
    d: int,
    problem_seed: int,
    *,
    iter_index_drivers: Callable[[str], tuple[EnnIncrementalIndexDriver, ...]],
    normalize_function_name: Callable[[str], str],
) -> Iterable[tuple[str, tuple[int, str, int, int, int, str]]]:
    shared = enn_batch_shared_params(num_reps=num_reps, d=d, problem_seed=problem_seed)
    d_i, ps_i, nr, chk = shared.d, shared.problem_seed, shared.num_reps, shared.checkpoint_ns
    drvs = iter_index_drivers(index_driver)
    for fm in map(normalize_function_name, shared.benchmark_functions):
        for drv in drvs:
            for ri in range(nr):
                dest = fit_ind_result_json_dest(
                    output_dir,
                    d=d_i,
                    function_name=fm,
                    problem_seed=ps_i,
                    rep_index=ri,
                    num_reps=nr,
                    index_driver=drv,
                    normalize_function_name=normalize_function_name,
                )
                if _fit_ind_json.fit_ind_result_json_complete(
                    dest,
                    chk,
                    d=d_i,
                    function_name=fm,
                    problem_seed=ps_i,
                    rep_index=ri,
                    num_reps=nr,
                    index_driver=drv,
                ):
                    continue
                yield (
                    fit_ind_job_key(
                        d=d_i,
                        function_name=fm,
                        problem_seed=ps_i,
                        rep_index=ri,
                        num_reps=nr,
                        index_driver=drv,
                        normalize_function_name=normalize_function_name,
                    ),
                    (d_i, fm, ps_i, ri, nr, drv.value),
                )
