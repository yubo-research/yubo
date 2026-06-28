"""ENN ``enn_fit()`` wall-time on synthetic benchmark draws (model build untimed)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from enn.enn.enn_class import EpistemicNearestNeighbors

from optimizer.uhd_enn_fit_helpers import fit_enn_params

from .batch_jobs import job_fit_quality
from .evaluate_metrics import normalize_benchmark_function_name
from .fitting_time import _SYNTHETIC_OBS_VAR, enn_fit_k_and_num_fit_samples
from .fitting_time_enn_incremental import (
    EnnIncrementalIndexDriver,
    enn_disk_work_dir,
    enn_test_log_likelihood,
    epistemic_nn_driver_kwargs,
)
from .fitting_time_enn_incremental_draw import _train_xy_unit_cube_segment


@dataclass(frozen=True)
class EnnFitTimingResult:
    n: int
    fit_seconds: float
    log_likelihood: float
    target: str
    d: int
    problem_seed: int
    data_seed: int
    index_driver: EnnIncrementalIndexDriver


def enn_fit_quality_ns() -> tuple[int, ...]:
    seen: list[int] = []
    for job in job_fit_quality():
        ni = int(job.n)
        if ni not in seen:
            seen.append(ni)
    return tuple(seen)


def benchmark_enn_fit_timing(
    *,
    D: int,
    function_name: str,
    data_seed: int,
    n: int,
    problem_seed: int,
    index_driver: EnnIncrementalIndexDriver = EnnIncrementalIndexDriver.FLAT,
) -> EnnFitTimingResult:
    target = normalize_benchmark_function_name(function_name)
    d = int(D)
    n_obs = int(n)
    if n_obs < 1:
        raise ValueError(f"n must be >= 1, got {n_obs}")
    draw_seed = int(data_seed)
    base_seed = int(problem_seed)

    train_x, train_y = _train_xy_unit_cube_segment(
        D=d,
        function_name=target,
        problem_seed=draw_seed,
        n_train=n_obs,
        start_row=0,
    )
    train_yvar = np.full_like(train_y, _SYNTHETIC_OBS_VAR)
    k_eff, nfs = enn_fit_k_and_num_fit_samples(n_obs)
    gen = np.random.default_rng(draw_seed)

    with enn_disk_work_dir(index_driver) as work_dir:
        enn_model = EpistemicNearestNeighbors(
            train_x,
            train_y,
            train_yvar,
            **epistemic_nn_driver_kwargs(index_driver, work_dir=work_dir),
        )
        t_0 = time.perf_counter()
        fit_enn_params(
            enn_model,
            train_x,
            train_y,
            k=k_eff,
            num_fit_candidates=100,
            num_fit_samples=nfs,
            rng=gen,
            yvar=train_yvar,
        )
        fit_seconds = time.perf_counter() - t_0
        log_likelihood = enn_test_log_likelihood(
            enn_model,
            D=d,
            function_name=target,
            problem_seed=draw_seed,
            n_obs=n_obs,
        )

    return EnnFitTimingResult(
        n=n_obs,
        fit_seconds=float(fit_seconds),
        log_likelihood=float(log_likelihood),
        target=target,
        d=d,
        problem_seed=base_seed,
        data_seed=draw_seed,
        index_driver=index_driver,
    )
