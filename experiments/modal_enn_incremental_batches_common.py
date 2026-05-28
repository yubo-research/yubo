"""Shared ENN incremental Modal batch control-plane logic (no worker image)."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Iterable

import modal

from analysis.fitting_time.evaluate_metrics import normalize_benchmark_function_name
from analysis.fitting_time.fitting_time_enn_incremental import (
    EnnIncrementalIndexDriver,
    EnnIncrementalTimingResult,
)
from experiments import modal_enn_fit_batches as _fit_batches
from experiments import modal_enn_fit_ind_batches as _fit_ind_batches
from experiments import modal_enn_full_opt_batches as _full_opt_batches
from experiments import modal_enn_incremental_batches_json as _add_json
from experiments import modal_enn_query_batches as _query_batches
from experiments.enn_batch_job_params import (
    enn_batch_shared_params,
    normalize_index_driver,
)

RunFn = Callable[..., subprocess.CompletedProcess]


def get_app_name(tag: str) -> str:
    return f"yubo-enn-incremental-{tag}"


def dict_names_for_tag(tag: str) -> tuple[str, str]:
    return (
        f"enn_incremental_results_{tag}",
        f"enn_incremental_submitted_{tag}",
    )


def experiment_type_from_tag(tag: str) -> str:
    prefix = tag.split("-", 1)[0]
    if prefix not in {"add_method", "fit_method", "fit_ind", "query", "full_optimization"}:
        raise ValueError(
            f"unknown experiment prefix {prefix!r} in tag {tag!r}; expected add_method-*, fit_method-*, fit_ind-*, query-*, or full_optimization-*"
        )
    return prefix


def results_dict(tag: str):
    return modal.Dict.from_name(f"enn_incremental_results_{tag}", create_if_missing=True)


def submitted_dict(tag: str):
    return modal.Dict.from_name(f"enn_incremental_submitted_{tag}", create_if_missing=True)


def iter_index_drivers(index_driver: str) -> tuple[EnnIncrementalIndexDriver, ...]:
    raw = str(index_driver).strip().lower().replace("-", "_")
    return tuple(EnnIncrementalIndexDriver) if raw == "all" else (normalize_index_driver(raw),)


def job_key(
    *,
    d: int,
    function_name: str,
    problem_seed: int,
    rep_index: int,
    num_reps: int,
    index_driver: str | EnnIncrementalIndexDriver,
) -> str:
    drv = normalize_index_driver(index_driver).value
    fn = normalize_benchmark_function_name(function_name)
    return f"enn_incremental_D{int(d)}_{fn}_pseed{int(problem_seed)}_nrep{int(num_reps)}_rep{int(rep_index)}_{drv}"


def result_json_dest(
    output_dir: str | Path,
    *,
    d: int,
    function_name: str,
    problem_seed: int,
    rep_index: int,
    num_reps: int,
    index_driver: str | EnnIncrementalIndexDriver,
) -> Path:
    return Path(output_dir) / (
        f"{job_key(d=d, function_name=function_name, problem_seed=problem_seed, rep_index=rep_index, num_reps=num_reps, index_driver=index_driver)}.json"
    )


def result_to_payload(
    result: EnnIncrementalTimingResult,
    *,
    problem_seed: int | None = None,
    data_seed: int | None = None,
    rep_index: int | None = None,
    num_reps: int | None = None,
) -> dict:
    meta = {
        "D": int(result.d),
        "function_name": result.target,
        "problem_seed": int(result.problem_seed if problem_seed is None else problem_seed),
        "index_driver": result.index_driver.value,
    }
    if data_seed is not None:
        meta["data_seed"] = int(data_seed)
    if rep_index is not None:
        meta["rep_index"] = int(rep_index)
    if num_reps is not None:
        meta["num_reps"] = int(num_reps)
    return {
        "N": list(result.n),
        "add_seconds": list(result.add_seconds),
        "log_likelihood": list(result.log_likelihood),
        "_meta": meta,
    }


def write_json(dest: str | Path, payload: dict) -> None:
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def result_json_complete(
    dest: str | Path,
    expected_n: tuple[int, ...],
    *,
    d: int,
    function_name: str,
    problem_seed: int,
    rep_index: int,
    num_reps: int,
    index_driver: str | EnnIncrementalIndexDriver,
) -> bool:
    return _add_json.result_json_complete(
        dest,
        expected_n,
        d=d,
        function_name=function_name,
        problem_seed=problem_seed,
        rep_index=rep_index,
        num_reps=num_reps,
        index_driver=index_driver,
    )


def pending_jobs(kind: str, output_dir: str | Path, index_driver: str, num_reps: int, d: int, ps: int):
    shared = enn_batch_shared_params(num_reps=num_reps, d=d, problem_seed=ps)
    drvs = iter_index_drivers(index_driver)
    d_i, ps_i, nr, chk = (
        shared.d,
        shared.problem_seed,
        shared.num_reps,
        shared.checkpoint_ns,
    )
    if kind == "add_method":
        for fm in map(normalize_benchmark_function_name, shared.benchmark_functions):
            for drv in drvs:
                for ri in range(nr):
                    dest = result_json_dest(
                        output_dir,
                        d=d_i,
                        function_name=fm,
                        problem_seed=ps_i,
                        rep_index=ri,
                        num_reps=nr,
                        index_driver=drv,
                    )
                    if result_json_complete(
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
                        job_key(
                            d=d_i,
                            function_name=fm,
                            problem_seed=ps_i,
                            rep_index=ri,
                            num_reps=nr,
                            index_driver=drv,
                        ),
                        (d_i, fm, ps_i, ri, nr, drv.value),
                    )
        return
    raise ValueError(f"unknown job kind {kind!r}; expected add_method")


def iter_incremental_jobs(
    output_dir: str | Path,
    index_driver: str,
    num_reps: int,
    d: int,
    problem_seed: int,
) -> Iterable[tuple[str, tuple[int, str, int, int, int, str]]]:
    yield from pending_jobs("add_method", output_dir, index_driver, num_reps, d, problem_seed)


def iter_fit_jobs(
    output_dir: str | Path,
    index_driver: str,
    num_reps: int,
    d: int,
    problem_seed: int,
) -> Iterable[tuple[str, tuple[int, str, int, int, int, int, str]]]:
    yield from _fit_batches.iter_fit_jobs(
        output_dir,
        index_driver,
        num_reps,
        d,
        problem_seed,
        iter_index_drivers=iter_index_drivers,
        normalize_function_name=normalize_benchmark_function_name,
    )


def iter_fit_ind_jobs(
    output_dir: str | Path,
    index_driver: str,
    num_reps: int,
    d: int,
    problem_seed: int,
) -> Iterable[tuple[str, tuple[int, str, int, int, int, str]]]:
    yield from _fit_ind_batches.iter_fit_ind_jobs(
        output_dir,
        index_driver,
        num_reps,
        d,
        problem_seed,
        iter_index_drivers=iter_index_drivers,
        normalize_function_name=normalize_benchmark_function_name,
    )


def iter_query_jobs(
    output_dir: str | Path,
    index_driver: str,
    num_reps: int,
    d: int,
    problem_seed: int,
) -> Iterable[tuple[str, tuple[int, str, int, int, int, str]]]:
    yield from _query_batches.iter_query_jobs(
        output_dir,
        index_driver,
        num_reps,
        d,
        problem_seed,
        iter_index_drivers=iter_index_drivers,
        normalize_function_name=normalize_benchmark_function_name,
    )


def iter_full_opt_jobs(
    output_dir: str | Path,
    index_driver: str,
    num_reps: int,
    d: int,
    problem_seed: int,
) -> Iterable[tuple[str, tuple[str, int, int, int, str]]]:
    del d, problem_seed
    yield from _full_opt_batches.iter_full_opt_jobs(
        output_dir,
        index_driver,
        num_reps,
        iter_index_drivers=iter_index_drivers,
    )


def submit_missing(
    tag: str,
    output_dir: str | Path,
    index_driver: str,
    num_reps: int,
    d: int,
    problem_seed: int,
    *,
    force: bool = False,
):
    exp = experiment_type_from_tag(tag)
    if exp == "fit_method":
        it = iter_fit_jobs
    elif exp == "fit_ind":
        it = iter_fit_ind_jobs
    elif exp == "query":
        it = iter_query_jobs
    elif exp == "full_optimization":
        it = iter_full_opt_jobs
        print("full_optimization: problem_seed=18+rep_index per replicate; --problem-seed and -d are ignored")
    else:
        it = iter_incremental_jobs
    submitted = submitted_dict(tag)
    batch, count = [], 0

    def flush():
        nonlocal batch
        if batch:
            modal.Function.from_name(get_app_name(tag), "enn_incremental_batch_submitter").spawn(batch, tag, force)
            batch.clear()

    for key, job in it(output_dir, index_driver, int(num_reps), int(d), int(problem_seed)):
        if (not force) and (key in submitted):
            continue
        batch.append((key, job))
        count += 1
        if len(batch) >= 1000:
            flush()
    flush()
    print(f"submitted {count} ENN batch jobs")


def collect(tag: str, output_dir: str | Path):
    exp = experiment_type_from_tag(tag)
    results = results_dict(tag)
    keys_out = []
    outp = Path(output_dir)
    del_fn = modal.Function.from_name(get_app_name(tag), "enn_incremental_batch_deleter")
    for key, payload in results.items():
        if isinstance(payload, dict):
            dest = outp / f"{key}.json"
            write_json(dest, payload)
            print(f"wrote {dest.resolve()}")
            keys_out.append(key)
            continue
        if exp == "fit_method":
            if len(payload) != 8:
                raise ValueError(f"bad fit Modal payload len={len(payload)} key={key!r}")
            rp, d, fm, n, pseed, ri, nr, idrv = payload
            dest = _fit_batches.fit_result_json_dest(
                outp,
                d=d,
                function_name=fm,
                n=int(n),
                problem_seed=pseed,
                rep_index=ri,
                num_reps=nr,
                index_driver=idrv,
                normalize_function_name=normalize_benchmark_function_name,
            )
        elif exp == "fit_ind":
            if len(payload) != 7:
                raise ValueError(f"bad fit_ind Modal payload len={len(payload)} key={key!r}")
            rp, d, fm, pseed, ri, nr, idrv = payload
            dest = _fit_ind_batches.fit_ind_result_json_dest(
                outp,
                d=d,
                function_name=fm,
                problem_seed=pseed,
                rep_index=ri,
                num_reps=nr,
                index_driver=idrv,
                normalize_function_name=normalize_benchmark_function_name,
            )
        elif exp == "query":
            if len(payload) != 7:
                raise ValueError(f"bad query Modal payload len={len(payload)} key={key!r}")
            rp, d, fm, pseed, ri, nr, idrv = payload
            dest = _query_batches.query_result_json_dest(
                outp,
                d=d,
                function_name=fm,
                problem_seed=pseed,
                rep_index=ri,
                num_reps=nr,
                index_driver=idrv,
                normalize_function_name=normalize_benchmark_function_name,
            )
        else:
            if len(payload) != 7:
                raise ValueError(f"bad add Modal payload len={len(payload)} key={key!r}")
            rp, d, fm, pseed, ri, nr, idrv = payload
            dest = result_json_dest(
                outp,
                d=d,
                function_name=fm,
                problem_seed=pseed,
                rep_index=ri,
                num_reps=nr,
                index_driver=idrv,
            )
        write_json(dest, rp)
        print(f"wrote {dest.resolve()}")
        keys_out.append(key)
    if keys_out:
        del_fn.spawn(keys_out, tag)
    print(f"collected {len(keys_out)} jobs")


def status(tag: str):
    print(f"results_available = {results_dict(tag).len()}")
    print(f"submitted = {submitted_dict(tag).len()}")


def clean_up(tag: str, *, run: RunFn | None = None) -> None:
    run_fn = subprocess.run if run is None else run
    exit_code = 0
    for name in dict_names_for_tag(tag):
        del_result = run_fn(["modal", "dict", "delete", "--yes", "--allow-missing", name])
        if del_result.returncode != 0:
            print(f"CLEANUP: dict delete failed name={name} returncode={del_result.returncode}")
            exit_code = 1
        else:
            print(f"CLEANUP: deleted dict name={name}")
    if exit_code != 0:
        sys.exit(exit_code)


def _stop_modal_app(app_name: str, *, run: RunFn | None = None) -> int:
    run_fn = subprocess.run if run is None else run
    result = run_fn(["modal", "app", "stop", app_name])
    return result.returncode


def stop(tag: str, *, run: RunFn | None = None) -> None:
    app_rc = _stop_modal_app(get_app_name(tag), run=run)
    clean_up(tag, run=run)
    if app_rc != 0:
        sys.exit(app_rc)


def run_command(
    tag: str,
    cmd: str,
    *,
    output_dir: str = "results/enn_incremental",
    index_driver: str = "all",
    num_reps: int = 10,
    d: int = 10,
    problem_seed: int = 17,
) -> None:
    if cmd == "submit":
        submit_missing(tag, output_dir, index_driver, int(num_reps), int(d), int(problem_seed))
    elif cmd == "submit-force":
        submit_missing(
            tag,
            output_dir,
            index_driver,
            int(num_reps),
            int(d),
            int(problem_seed),
            force=True,
        )
    elif cmd == "collect":
        collect(tag, output_dir)
    elif cmd == "status":
        status(tag)
    elif cmd == "clean_up":
        clean_up(tag)
    elif cmd == "stop":
        stop(tag)
    else:
        raise ValueError(f"Unknown command: {cmd}")
