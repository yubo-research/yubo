"""Modal worker dispatch for ENN incremental batch jobs."""

from __future__ import annotations

from typing import Callable

from analysis.fitting_time import benchmark_enn_incremental_add_timing
from analysis.fitting_time.evaluate import synthetic_benchmark_data_seed
from analysis.fitting_time.evaluate_metrics import normalize_benchmark_function_name
from analysis.fitting_time.fitting_time_enn_fit import benchmark_enn_fit_timing
from analysis.fitting_time.fitting_time_enn_fit_ind import benchmark_enn_fit_ind_timing
from experiments import modal_enn_fit_batches as _fit_batches
from experiments import modal_enn_fit_ind_batches as _fit_ind_batches
from experiments.enn_batch_job_params import normalize_index_driver


def dispatch_enn_incremental_batch_worker(
    job,
    *,
    experiment_type_from_tag: Callable[[str], str],
    job_key,
    result_to_payload,
    results_dict: Callable[[str], object],
) -> None:
    lj = len(job)
    if lj < 2:
        raise ValueError(f"expected job with tag; got len={lj}")
    tag = job[0]
    exp = experiment_type_from_tag(tag)
    if exp == "add_method":
        if lj != 7:
            raise ValueError(f"add_method job expected 7 fields after tag; got len={lj}")
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
        val = (
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
        results_dict(tag)[ky] = val
        return
    if exp == "fit_ind":
        if lj != 7:
            raise ValueError(f"fit_ind job expected 7 fields after tag; got len={lj}")
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
        val = (
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
        results_dict(tag)[ky] = val
        return
    if exp != "fit_method":
        raise ValueError(f"unknown experiment {exp!r} in tag {tag!r}")
    if lj != 8:
        raise ValueError(f"fit_method job expected 8 fields after tag; got len={lj}")
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
    val = (
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
    results_dict(tag)[ky] = val
