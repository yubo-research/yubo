"""ENN posterior query timing on synthetic benchmark draws."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .evaluate_metrics import normalize_benchmark_function_name
from .fitting_time import _SYNTHETIC_OBS_VAR
from .fitting_time_enn_incremental import (
    EnnIncrementalIndexDriver,
    _checkpoint_enn_params,
    enn_disk_work_dir,
    enn_incremental_checkpoint_ns,
    make_epistemic_nn,
    sync_enn_index,
)
from .fitting_time_enn_incremental_draw import (
    _train_xy_unit_cube_segment,
    draw_benchmark_test_xy_unit_cube,
)

ENN_QUERY_NUM_POINTS = 100


@dataclass(frozen=True)
class EnnQueryTimingResult:
    n: tuple[int, ...]
    query_seconds: tuple[float, ...]
    query_seconds_per_point: tuple[float, ...]
    num_query_points: int
    target: str
    d: int
    problem_seed: int
    index_driver: EnnIncrementalIndexDriver


def benchmark_enn_query_timing(
    *,
    D: int,
    function_name: str,
    problem_seed: int,
    index_driver: EnnIncrementalIndexDriver = EnnIncrementalIndexDriver.FLAT,
    checkpoints: Sequence[int] | None = None,
    num_query_points: int = ENN_QUERY_NUM_POINTS,
) -> EnnQueryTimingResult:
    from enn.enn.enn_params import PosteriorFlags

    target = normalize_benchmark_function_name(function_name)
    d = int(D)
    seed = int(problem_seed)
    n_query = int(num_query_points)
    if n_query < 1:
        raise ValueError(f"num_query_points must be >= 1, got {n_query}")
    ckpts = tuple(checkpoints) if checkpoints is not None else enn_incremental_checkpoint_ns()
    if len(ckpts) == 0:
        raise ValueError("checkpoints must be non-empty")
    prev_n = 0
    for n_chk in ckpts:
        if int(n_chk) <= prev_n:
            raise ValueError(f"checkpoints must be strictly increasing, got {ckpts}")
        prev_n = int(n_chk)

    x_query, _ = draw_benchmark_test_xy_unit_cube(D=d, function_name=target, problem_seed=seed)
    x_query = np.asarray(x_query[:n_query], dtype=np.float64)
    if int(x_query.shape[0]) != n_query:
        raise ValueError(f"requested {n_query} query points but only drew {x_query.shape[0]}")

    yvar_scalar = float(_SYNTHETIC_OBS_VAR)
    flags = PosteriorFlags(observation_noise=False)

    ns: list[int] = []
    query_seconds: list[float] = []
    query_seconds_per_point: list[float] = []

    with enn_disk_work_dir(index_driver) as work_dir:
        for n_chk in ckpts:
            n_obs = int(n_chk)
            train_x, train_y = _train_xy_unit_cube_segment(
                D=d,
                function_name=target,
                problem_seed=seed,
                n_train=n_obs,
                start_row=0,
            )
            enn_model = make_epistemic_nn(
                train_x,
                train_y,
                np.full_like(train_y, yvar_scalar),
                index_driver,
                work_dir=work_dir,
            )
            params = _checkpoint_enn_params(n_obs)
            sync_enn_index(enn_model)
            t_0 = time.perf_counter()
            enn_model.posterior(x_query, params=params, flags=flags)
            elapsed = time.perf_counter() - t_0
            ns.append(n_obs)
            query_seconds.append(float(elapsed))
            query_seconds_per_point.append(float(elapsed) / float(n_query))

    return EnnQueryTimingResult(
        n=tuple(ns),
        query_seconds=tuple(query_seconds),
        query_seconds_per_point=tuple(query_seconds_per_point),
        num_query_points=n_query,
        target=target,
        d=d,
        problem_seed=seed,
        index_driver=index_driver,
    )
