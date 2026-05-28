"""Fit-ind job keys, JSON payloads, and pending-job iteration for ENN Modal batches."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from analysis.fitting_time.fitting_time_enn_fit_ind import EnnFitIndTimingResult
from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver
from experiments import modal_enn_fit_ind_batches_json as _fit_ind_json
from experiments.enn_batch_job_params import normalize_index_driver
from experiments.modal_enn_series_batches import iter_replicate_series_jobs


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
    yield from iter_replicate_series_jobs(
        output_dir,
        index_driver,
        num_reps,
        d,
        problem_seed,
        iter_index_drivers=iter_index_drivers,
        normalize_function_name=normalize_function_name,
        result_json_dest=fit_ind_result_json_dest,
        result_json_complete=_fit_ind_json.fit_ind_result_json_complete,
        job_key=fit_ind_job_key,
    )
