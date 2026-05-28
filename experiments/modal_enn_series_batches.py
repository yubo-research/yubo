"""Shared helpers for per-replicate ENN batch result series."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver
from experiments.enn_batch_job_params import enn_batch_shared_params


def iter_replicate_series_jobs(
    output_dir: str | Path,
    index_driver: str,
    num_reps: int,
    d: int,
    problem_seed: int,
    *,
    iter_index_drivers: Callable[[str], tuple[EnnIncrementalIndexDriver, ...]],
    normalize_function_name: Callable[[str], str],
    result_json_dest: Callable[..., Path],
    result_json_complete: Callable[..., bool],
    job_key: Callable[..., str],
) -> Iterable[tuple[str, tuple[int, str, int, int, int, str]]]:
    shared = enn_batch_shared_params(num_reps=num_reps, d=d, problem_seed=problem_seed)
    d_i, ps_i, nr, chk = (
        shared.d,
        shared.problem_seed,
        shared.num_reps,
        shared.checkpoint_ns,
    )
    for fm in map(normalize_function_name, shared.benchmark_functions):
        for drv in iter_index_drivers(index_driver):
            for ri in range(nr):
                dest = result_json_dest(
                    output_dir,
                    d=d_i,
                    function_name=fm,
                    problem_seed=ps_i,
                    rep_index=ri,
                    num_reps=nr,
                    index_driver=drv,
                    normalize_function_name=normalize_function_name,
                )
                complete = result_json_complete(
                    dest,
                    chk,
                    d=d_i,
                    function_name=fm,
                    problem_seed=ps_i,
                    rep_index=ri,
                    num_reps=nr,
                    index_driver=drv,
                )
                if complete:
                    continue
                key = job_key(
                    d=d_i,
                    function_name=fm,
                    problem_seed=ps_i,
                    rep_index=ri,
                    num_reps=nr,
                    index_driver=drv,
                    normalize_function_name=normalize_function_name,
                )
                yield key, (d_i, fm, ps_i, ri, nr, drv.value)
