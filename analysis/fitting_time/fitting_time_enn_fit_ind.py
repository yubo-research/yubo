"""Incremental ENN per-add ``enn_fit()`` timing with warm-started parameters."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .evaluate_metrics import normalize_benchmark_function_name
from .fitting_time import _SYNTHETIC_OBS_VAR, enn_fit_k_and_num_fit_samples
from .fitting_time_enn_incremental import (
    EnnIncrementalIndexDriver,
    enn_incremental_checkpoint_ns,
    enn_test_log_likelihood,
)
from .fitting_time_enn_incremental_draw import _train_xy_unit_cube_segment


@dataclass(frozen=True)
class EnnFitIndTimingResult:
    n: tuple[int, ...]
    fit_seconds: tuple[float, ...]
    log_likelihood: tuple[float, ...]
    target: str
    d: int
    problem_seed: int
    index_driver: EnnIncrementalIndexDriver


def _enn_fit_timed_after_add(enn_model, *, current_n: int, rng, params_warm_start):
    from enn.enn.enn_fit import enn_fit

    k_eff, nfs = enn_fit_k_and_num_fit_samples(int(current_n))
    enn_fit(
        enn_model,
        k=k_eff,
        num_fit_candidates=1,
        num_fit_samples=nfs,
        rng=rng,
        params_warm_start=params_warm_start,
    )
    t_0 = time.perf_counter()
    params = enn_fit(
        enn_model,
        k=k_eff,
        num_fit_candidates=1,
        num_fit_samples=nfs,
        rng=rng,
        params_warm_start=params_warm_start,
    )
    return params, time.perf_counter() - t_0


def _fit_probability_after_add(current_n: int) -> float:
    _, nfs = enn_fit_k_and_num_fit_samples(int(current_n))
    return min(1.0, float(nfs) / float(current_n))


def _add_segment_with_per_point_fit(
    enn_model,
    *,
    x_seg: np.ndarray,
    y_seg: np.ndarray,
    yvar_row: np.ndarray,
    start_n: int,
    rng,
    params_warm_start,
) -> tuple[object, float]:
    fit_total = 0.0
    params = params_warm_start
    n_rows = int(x_seg.shape[0])
    for i in range(n_rows):
        enn_model.add(x_seg[i : i + 1], y_seg[i : i + 1], yvar_row)
        current_n = int(start_n) + i + 1
        fit_p = _fit_probability_after_add(current_n)
        if fit_p < 1.0 and float(rng.random()) >= fit_p:
            continue
        params, dt = _enn_fit_timed_after_add(
            enn_model,
            current_n=current_n,
            rng=rng,
            params_warm_start=params,
        )
        fit_total += dt
    return params, fit_total


def benchmark_enn_fit_ind_timing(
    *,
    D: int,
    function_name: str,
    problem_seed: int,
    index_driver: EnnIncrementalIndexDriver = EnnIncrementalIndexDriver.FLAT,
    checkpoints: Sequence[int] | None = None,
) -> EnnFitIndTimingResult:
    from enn.enn.enn_class import EpistemicNearestNeighbors

    target = normalize_benchmark_function_name(function_name)
    d = int(D)
    seed = int(problem_seed)
    ckpts = tuple(checkpoints) if checkpoints is not None else enn_incremental_checkpoint_ns()
    if len(ckpts) == 0:
        raise ValueError("checkpoints must be non-empty")
    prev_n = 0
    for n_chk in ckpts:
        if int(n_chk) <= prev_n:
            raise ValueError(f"checkpoints must be strictly increasing, got {ckpts}")

    driver = index_driver.to_enn_index_driver()
    enn_model = EpistemicNearestNeighbors(
        np.zeros((0, d), dtype=np.float64),
        np.zeros((0, 1), dtype=np.float64),
        index_driver=driver,
    )
    yvar_row = np.array([[float(_SYNTHETIC_OBS_VAR)]], dtype=np.float64)
    rng = np.random.default_rng(seed)

    ns: list[int] = []
    fit_seconds: list[float] = []
    log_likelihood: list[float] = []
    params_warm_start = None

    for n_chk in ckpts:
        n_target = int(n_chk)
        x_seg, y_seg = _train_xy_unit_cube_segment(
            D=d,
            function_name=target,
            problem_seed=seed,
            n_train=n_target,
            start_row=prev_n,
        )
        params_warm_start, seg_fit_seconds = _add_segment_with_per_point_fit(
            enn_model,
            x_seg=x_seg,
            y_seg=y_seg,
            yvar_row=yvar_row,
            start_n=prev_n,
            rng=rng,
            params_warm_start=params_warm_start,
        )
        fit_seconds.append(seg_fit_seconds)
        prev_n = n_target
        log_likelihood.append(
            enn_test_log_likelihood(
                enn_model,
                D=d,
                function_name=target,
                problem_seed=seed,
                n_obs=n_target,
            )
        )
        ns.append(n_target)

    return EnnFitIndTimingResult(
        n=tuple(ns),
        fit_seconds=tuple(fit_seconds),
        log_likelihood=tuple(log_likelihood),
        target=target,
        d=d,
        problem_seed=seed,
        index_driver=index_driver,
    )
