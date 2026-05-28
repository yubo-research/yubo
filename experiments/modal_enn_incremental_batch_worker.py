"""Modal worker dispatch for ENN incremental batch jobs."""

from __future__ import annotations

from typing import Callable

from analysis.fitting_time import benchmark_enn_incremental_add_timing
from analysis.fitting_time.evaluate import synthetic_benchmark_data_seed
from analysis.fitting_time.evaluate_metrics import normalize_benchmark_function_name
from analysis.fitting_time.fitting_time_enn_fit import benchmark_enn_fit_timing
from analysis.fitting_time.fitting_time_enn_fit_ind import benchmark_enn_fit_ind_timing
from analysis.fitting_time.fitting_time_enn_full_opt import benchmark_enn_full_optimization_proposal_timing
from analysis.fitting_time.fitting_time_enn_query import benchmark_enn_query_timing
from experiments import modal_enn_fit_batches as _fit_batches
from experiments import modal_enn_fit_ind_batches as _fit_ind_batches
from experiments import modal_enn_full_opt_batches as _full_opt_batches
from experiments import modal_enn_query_batches as _query_batches
from experiments.enn_batch_job_params import normalize_index_driver

_JobHandler = Callable[..., None]


def _expect_len(job, n: int, label: str) -> None:
    lj = len(job)
    if lj != n:
        raise ValueError(f"{label} job expected {n} fields after tag; got len={lj}")


def _write_full_opt_result(
    *,
    tag: str,
    env_tag: str,
    problem_seed: int,
    rep_index: int,
    num_reps: int,
    index_driver,
    results_dict: Callable[[str], object],
) -> None:
    drv = normalize_index_driver(index_driver)
    ky = _full_opt_batches.full_opt_job_key(
        env_tag=str(env_tag),
        problem_seed=int(problem_seed),
        rep_index=int(rep_index),
        num_reps=int(num_reps),
        index_driver=drv,
    )
    result = benchmark_enn_full_optimization_proposal_timing(
        env_tag=str(env_tag),
        problem_seed=int(problem_seed),
        rep_index=int(rep_index),
        index_driver=drv,
    )
    results_dict(tag)[ky] = _full_opt_batches.full_opt_result_to_payload(
        result,
        num_reps=int(num_reps),
    )


def _write_query_result(
    *,
    tag: str,
    d: int,
    function_name: str,
    problem_seed: int,
    rep_index: int,
    num_reps: int,
    index_driver,
    results_dict: Callable[[str], object],
) -> None:
    drv = normalize_index_driver(index_driver)
    ds = synthetic_benchmark_data_seed(
        function_name=function_name,
        problem_seed=int(problem_seed),
        rep_index=int(rep_index),
    )
    result = benchmark_enn_query_timing(
        D=int(d),
        function_name=function_name,
        problem_seed=ds,
        index_driver=drv,
    )
    ky = _query_batches.query_job_key(
        d=int(d),
        function_name=function_name,
        problem_seed=int(problem_seed),
        rep_index=int(rep_index),
        num_reps=int(num_reps),
        index_driver=drv,
        normalize_function_name=normalize_benchmark_function_name,
    )
    val = (
        _query_batches.query_result_to_payload(
            result,
            problem_seed=int(problem_seed),
            data_seed=ds,
            rep_index=int(rep_index),
            num_reps=int(num_reps),
        ),
        int(d),
        result.target,
        int(problem_seed),
        int(rep_index),
        int(num_reps),
        drv.value,
    )
    results_dict(tag)[ky] = val


def _handle_add_method(job, *, tag, job_key, result_to_payload, results_dict) -> None:
    _expect_len(job, 7, "add_method")
    _, d, function_name, problem_seed, rep_index, num_reps, index_driver = job
    drv = normalize_index_driver(index_driver)
    ds = synthetic_benchmark_data_seed(
        function_name=function_name,
        problem_seed=int(problem_seed),
        rep_index=int(rep_index),
    )
    result = benchmark_enn_incremental_add_timing(
        D=int(d),
        function_name=function_name,
        problem_seed=ds,
        index_driver=drv,
    )
    ky = job_key(
        d=int(d),
        function_name=function_name,
        problem_seed=int(problem_seed),
        rep_index=int(rep_index),
        num_reps=int(num_reps),
        index_driver=drv,
    )
    results_dict(tag)[ky] = (
        result_to_payload(
            result,
            problem_seed=int(problem_seed),
            data_seed=ds,
            rep_index=int(rep_index),
            num_reps=int(num_reps),
        ),
        int(d),
        result.target,
        int(problem_seed),
        int(rep_index),
        int(num_reps),
        drv.value,
    )


def _handle_fit_ind(job, *, tag, job_key, result_to_payload, results_dict) -> None:
    del job_key, result_to_payload
    _expect_len(job, 7, "fit_ind")
    _, d, function_name, problem_seed, rep_index, num_reps, index_driver = job
    drv = normalize_index_driver(index_driver)
    ds = synthetic_benchmark_data_seed(
        function_name=function_name,
        problem_seed=int(problem_seed),
        rep_index=int(rep_index),
    )
    result = benchmark_enn_fit_ind_timing(
        D=int(d),
        function_name=function_name,
        problem_seed=ds,
        index_driver=drv,
    )
    ky = _fit_ind_batches.fit_ind_job_key(
        d=int(d),
        function_name=function_name,
        problem_seed=int(problem_seed),
        rep_index=int(rep_index),
        num_reps=int(num_reps),
        index_driver=drv,
        normalize_function_name=normalize_benchmark_function_name,
    )
    results_dict(tag)[ky] = (
        _fit_ind_batches.fit_ind_result_to_payload(
            result,
            problem_seed=int(problem_seed),
            data_seed=ds,
            rep_index=int(rep_index),
            num_reps=int(num_reps),
        ),
        int(d),
        result.target,
        int(problem_seed),
        int(rep_index),
        int(num_reps),
        drv.value,
    )


def _handle_query(job, *, tag, job_key, result_to_payload, results_dict) -> None:
    del job_key, result_to_payload
    _expect_len(job, 7, "query")
    _, d, function_name, problem_seed, rep_index, num_reps, index_driver = job
    _write_query_result(
        tag=tag,
        d=d,
        function_name=function_name,
        problem_seed=problem_seed,
        rep_index=rep_index,
        num_reps=num_reps,
        index_driver=index_driver,
        results_dict=results_dict,
    )


def _handle_full_optimization(job, *, tag, job_key, result_to_payload, results_dict) -> None:
    del job_key, result_to_payload
    _expect_len(job, 6, "full_optimization")
    _, env_tag, problem_seed, rep_index, num_reps, index_driver = job
    _write_full_opt_result(
        tag=tag,
        env_tag=env_tag,
        problem_seed=problem_seed,
        rep_index=rep_index,
        num_reps=num_reps,
        index_driver=index_driver,
        results_dict=results_dict,
    )


def _handle_fit_method(job, *, tag, job_key, result_to_payload, results_dict) -> None:
    del job_key, result_to_payload
    _expect_len(job, 8, "fit_method")
    _, d, function_name, n, problem_seed, rep_index, num_reps, index_driver = job
    drv = normalize_index_driver(index_driver)
    ds = synthetic_benchmark_data_seed(
        function_name=function_name,
        problem_seed=int(problem_seed),
        rep_index=int(rep_index),
    )
    result = benchmark_enn_fit_timing(
        D=int(d),
        function_name=function_name,
        data_seed=int(ds),
        problem_seed=int(problem_seed),
        n=int(n),
        index_driver=drv,
    )
    ky = _fit_batches.fit_job_key(
        d=int(d),
        function_name=function_name,
        n=int(n),
        problem_seed=int(problem_seed),
        rep_index=int(rep_index),
        num_reps=int(num_reps),
        index_driver=drv,
        normalize_function_name=normalize_benchmark_function_name,
    )
    results_dict(tag)[ky] = (
        _fit_batches.fit_result_to_payload(
            result,
            problem_seed=int(problem_seed),
            data_seed=int(ds),
            rep_index=int(rep_index),
            num_reps=int(num_reps),
        ),
        int(d),
        result.target,
        int(n),
        int(problem_seed),
        int(rep_index),
        int(num_reps),
        drv.value,
    )


_JOB_HANDLERS: dict[str, _JobHandler] = {
    "add_method": _handle_add_method,
    "fit_ind": _handle_fit_ind,
    "query": _handle_query,
    "full_optimization": _handle_full_optimization,
    "fit_method": _handle_fit_method,
}


def dispatch_enn_incremental_batch_worker(
    job,
    *,
    experiment_type_from_tag: Callable[[str], str],
    job_key,
    result_to_payload,
    results_dict: Callable[[str], object],
) -> None:
    if len(job) < 2:
        raise ValueError(f"expected job with tag; got len={len(job)}")
    tag = job[0]
    exp = experiment_type_from_tag(tag)
    handler = _JOB_HANDLERS.get(exp)
    if handler is None:
        raise ValueError(f"unknown experiment {exp!r} in tag {tag!r}")
    handler(
        job,
        tag=tag,
        job_key=job_key,
        result_to_payload=result_to_payload,
        results_dict=results_dict,
    )
