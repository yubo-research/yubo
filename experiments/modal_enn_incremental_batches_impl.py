"""Async Modal batch workflow for ENN add-timing and fit-timing experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import modal

from analysis.fitting_time import benchmark_enn_incremental_add_timing
from analysis.fitting_time.evaluate import synthetic_benchmark_data_seed
from analysis.fitting_time.evaluate_metrics import normalize_benchmark_function_name
from analysis.fitting_time.fitting_time_enn_fit import benchmark_enn_fit_timing
from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver, EnnIncrementalTimingResult
from experiments import modal_enn_fit_batches as _fit_batches
from experiments import modal_enn_incremental_batches_json as _add_json
from experiments.enn_batch_job_params import (
    enn_batch_shared_params,
    normalize_index_driver,
)
from experiments.modal_dict_utils import delete_keys_from_dicts
from experiments.modal_image import mk_image

_TAG = os.environ.get("MODAL_TAG", "add_method-default")
_modal_image = mk_image(_TAG)


def _get_app_name(tag: str) -> str:
    return f"yubo-enn-incremental-{tag}"


def _experiment_type_from_tag(tag: str) -> str:
    prefix = tag.split("-", 1)[0]
    if prefix not in {"add_method", "fit_method"}:
        raise ValueError(f"unknown experiment prefix {prefix!r} in tag {tag!r}; expected add_method-* or fit_method-*")
    return prefix


app = modal.App(name=_get_app_name(_TAG))


def _results_dict(tag: str):
    return modal.Dict.from_name(f"enn_incremental_results_{tag}", create_if_missing=True)


def _submitted_dict(tag: str):
    return modal.Dict.from_name(f"enn_incremental_submitted_{tag}", create_if_missing=True)


def _iter_index_drivers(index_driver: str) -> tuple[EnnIncrementalIndexDriver, ...]:
    raw = str(index_driver).strip().lower().replace("-", "_")
    return tuple(EnnIncrementalIndexDriver) if raw == "all" else (normalize_index_driver(raw),)


def _job_key(
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
        f"{_job_key(d=d, function_name=function_name, problem_seed=problem_seed, rep_index=rep_index, num_reps=num_reps, index_driver=index_driver)}.json"
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


def _write_json(dest: str | Path, payload: dict) -> None:
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _result_json_complete(
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


def _pending_jobs(kind: str, output_dir: str | Path, index_driver: str, num_reps: int, d: int, ps: int):
    shared = enn_batch_shared_params(num_reps=num_reps, d=d, problem_seed=ps)
    drvs = _iter_index_drivers(index_driver)
    d_i, ps_i, nr, chk = shared.d, shared.problem_seed, shared.num_reps, shared.checkpoint_ns
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
                    if _result_json_complete(
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
                        _job_key(
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


def _iter_incremental_jobs(
    output_dir: str | Path,
    index_driver: str,
    num_reps: int,
    d: int,
    problem_seed: int,
) -> Iterable[tuple[str, tuple[int, str, int, int, int, str]]]:
    yield from _pending_jobs("add_method", output_dir, index_driver, num_reps, d, problem_seed)


def _iter_fit_jobs(
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
        iter_index_drivers=_iter_index_drivers,
        normalize_function_name=normalize_benchmark_function_name,
    )


@app.function(image=_modal_image, max_containers=100, timeout=12 * 60 * 60, memory=4 * 1024, cpu=1.0)
def enn_incremental_batch_worker(job):
    lj = len(job)
    if lj < 2:
        raise ValueError(f"expected job with tag; got len={lj}")
    tag = job[0]
    exp = _experiment_type_from_tag(tag)
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
        ky = _job_key(
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
        _results_dict(tag)[ky] = val
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
    _results_dict(tag)[ky] = val


@app.function(image=_modal_image, max_containers=10, timeout=60 * 60)
def enn_incremental_batch_submitter(batch_of_jobs, tag: str, force: bool = False):
    submitted = _submitted_dict(tag)
    todo = []
    for key, job in batch_of_jobs:
        if (not force) and (key in submitted):
            continue
        submitted[key] = True
        todo.append((tag, *job))
    print(f"TODO: {len(todo)}")
    if todo:
        modal.Function.from_name(_get_app_name(tag), "enn_incremental_batch_worker").spawn_map(todo)


@app.function(image=_modal_image, max_containers=1, timeout=60 * 60)
def enn_incremental_batch_deleter(keys, tag: str):
    delete_keys_from_dicts(keys, _results_dict(tag), _submitted_dict(tag))


def _submit_missing(
    tag: str,
    output_dir: str | Path,
    index_driver: str,
    num_reps: int,
    d: int,
    problem_seed: int,
    *,
    force: bool = False,
):
    it = _iter_fit_jobs if _experiment_type_from_tag(tag) == "fit_method" else _iter_incremental_jobs
    submitted = _submitted_dict(tag)
    batch, count = [], 0

    def _flush():
        nonlocal batch
        if batch:
            modal.Function.from_name(_get_app_name(tag), "enn_incremental_batch_submitter").spawn(batch, tag, force)
            batch.clear()

    for key, job in it(output_dir, index_driver, int(num_reps), int(d), int(problem_seed)):
        if (not force) and (key in submitted):
            continue
        batch.append((key, job))
        count += 1
        if len(batch) >= 1000:
            _flush()
    _flush()
    print(f"submitted {count} ENN batch jobs")


def _collect(tag: str, output_dir: str | Path):
    fit = _experiment_type_from_tag(tag) == "fit_method"
    results = _results_dict(tag)
    keys_out = []
    outp = Path(output_dir)
    del_fn = modal.Function.from_name(_get_app_name(tag), "enn_incremental_batch_deleter")
    for key, payload in results.items():
        if isinstance(payload, dict):
            dest = outp / f"{key}.json"
            _write_json(dest, payload)
            print(f"wrote {dest.resolve()}")
            keys_out.append(key)
            continue
        if fit:
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
        else:
            if len(payload) != 7:
                raise ValueError(f"bad add Modal payload len={len(payload)} key={key!r}")
            rp, d, fm, pseed, ri, nr, idrv = payload
            dest = result_json_dest(outp, d=d, function_name=fm, problem_seed=pseed, rep_index=ri, num_reps=nr, index_driver=idrv)
        _write_json(dest, rp)
        print(f"wrote {dest.resolve()}")
        keys_out.append(key)
    if keys_out:
        del_fn.spawn(keys_out, tag)
    print(f"collected {len(keys_out)} jobs")


def status(tag: str):
    print(f"results_available = {_results_dict(tag).len()}")
    print(f"submitted = {_submitted_dict(tag).len()}")


def clean_up(tag: str):
    for name in (f"enn_incremental_results_{tag}", f"enn_incremental_submitted_{tag}"):
        try:
            modal.Dict.objects.delete(name, allow_missing=True)
            print(f"CLEANUP: deleted dict name={name}")
        except Exception as e:
            print(f"CLEANUP: dict delete failed name={name} err={e!r}")


def stop(tag: str):
    clean_up(tag)


@app.local_entrypoint()
def batches(
    tag: str,
    cmd: str,
    output_dir: str = "results/enn_incremental",
    index_driver: str = "all",
    num_reps: int = 10,
    d: int = 10,
    problem_seed: int = 17,
):
    if cmd == "submit":
        _submit_missing(tag, output_dir, index_driver, int(num_reps), int(d), int(problem_seed))
    elif cmd == "submit-force":
        _submit_missing(
            tag,
            output_dir,
            index_driver,
            int(num_reps),
            int(d),
            int(problem_seed),
            force=True,
        )
    elif cmd == "collect":
        _collect(tag, output_dir)
    elif cmd == "status":
        status(tag)
    elif cmd == "clean_up":
        clean_up(tag)
    elif cmd == "stop":
        stop(tag)
    else:
        raise ValueError(f"Unknown command: {cmd}")


__all__ = [
    "_collect",
    "_experiment_type_from_tag",
    "_get_app_name",
    "_iter_fit_jobs",
    "_iter_incremental_jobs",
    "_job_key",
    "normalize_index_driver",
    "_results_dict",
    "_submitted_dict",
    "_submit_missing",
    "app",
    "batches",
    "clean_up",
    "enn_incremental_batch_deleter",
    "enn_incremental_batch_submitter",
    "enn_incremental_batch_worker",
    "result_json_dest",
    "result_to_payload",
    "status",
    "stop",
]
