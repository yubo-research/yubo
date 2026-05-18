"""Shared experiment parameters for ENN add/fit Modal batch workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from analysis.fitting_time.evaluate import synthetic_benchmark_data_seed
from analysis.fitting_time.fitting_time_enn_incremental import (
    EnnIncrementalIndexDriver,
    enn_incremental_checkpoint_ns,
)

ENN_BATCH_BENCHMARK_FUNCTIONS: tuple[str, ...] = ("sphere", "ackley", "rosenbrock", "booth")


def normalize_index_driver(
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


def enn_batch_checkpoint_ns() -> tuple[int, ...]:
    return enn_incremental_checkpoint_ns()


def enn_batch_rep_meta_matches(
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
    drv = normalize_index_driver(index_driver).value
    fn = normalize_function_name(function_name)
    expected_data_seed = synthetic_benchmark_data_seed(
        function_name=fn,
        problem_seed=int(problem_seed),
        rep_index=int(rep_index),
    )
    checks = (
        int(meta["D"]) == int(d),
        normalize_function_name(str(meta["function_name"])) == fn,
        int(meta["problem_seed"]) == int(problem_seed),
        int(meta["data_seed"]) == expected_data_seed,
        int(meta["rep_index"]) == int(rep_index),
        int(meta["num_reps"]) == int(num_reps),
        str(meta["index_driver"]).strip().lower() == drv,
    )
    return all(checks)


def validate_enn_batch_scalars(*, num_reps: int, d: int) -> tuple[int, int]:
    nr = int(num_reps)
    d_i = int(d)
    if nr < 1:
        raise ValueError("num_reps must be >= 1")
    if d_i < 1:
        raise ValueError("D must be positive")
    return d_i, nr


@dataclass(frozen=True)
class EnnBatchSharedParams:
    benchmark_functions: tuple[str, ...]
    checkpoint_ns: tuple[int, ...]
    d: int
    problem_seed: int
    num_reps: int


def enn_batch_shared_params(
    *,
    num_reps: int,
    d: int,
    problem_seed: int,
) -> EnnBatchSharedParams:
    d_i, nr = validate_enn_batch_scalars(num_reps=num_reps, d=d)
    return EnnBatchSharedParams(
        benchmark_functions=ENN_BATCH_BENCHMARK_FUNCTIONS,
        checkpoint_ns=enn_batch_checkpoint_ns(),
        d=d_i,
        problem_seed=int(problem_seed),
        num_reps=nr,
    )
