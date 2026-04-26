"""Async Modal batch workflow for synthetic surrogate benchmarks.

Use via the ops wrapper in ``ops/synthetic_sine_benchmark_batches.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

from analysis.fitting_time.evaluate import synthetic_benchmark_data_seed
from experiments import synthetic_sine_benchmark_payload as ssbp
from experiments.modal_image import mk_image
from experiments.modal_synthetic_sine_benchmark_batches_reps import (
    aggregate_reps_to_dest,
    benchmark_json_dest,
    iter_missing_jobs,
    job_key,
    rep_json_dest,
)

_TAG = os.environ.get("MODAL_TAG")
if not _TAG:
    for i, arg in enumerate(sys.argv):
        if "modal_synthetic_sine_benchmark_batches_impl" in arg and i + 1 < len(sys.argv):
            candidate = sys.argv[i + 1]
            if not candidate.startswith("-"):
                _TAG = candidate
                break
    else:
        _TAG = "default"

_modal_image = mk_image(_TAG)


def _get_app_name(tag: str) -> str:
    return f"yubo-synth-sine-batch-{tag}"


_app_name = _get_app_name(_TAG)
app = modal.App(name=_app_name)


def _results_dict(tag: str):
    return modal.Dict.from_name(f"synthetic_sine_benchmark_results_{tag}", create_if_missing=True)


def _submitted_dict(tag: str):
    return modal.Dict.from_name(f"synthetic_sine_benchmark_submitted_{tag}", create_if_missing=True)


_job_key = job_key
_benchmark_json_dest = benchmark_json_dest
_rep_json_dest = rep_json_dest
_aggregate_reps_to_dest = aggregate_reps_to_dest
_iter_missing_jobs = iter_missing_jobs


@app.function(
    image=_modal_image,
    max_containers=1000,
    timeout=24 * 60 * 60,
    memory=2 * 1024,
    cpu=1.0,
)
def synthetic_sine_benchmark_batch_worker(job):
    tag, n, d, function_name, problem_seed, rep_index, num_reps = job
    key = _job_key(
        n=n,
        d=d,
        function_name=function_name,
        problem_seed=problem_seed,
        num_reps=num_reps,
    )
    data_seed = synthetic_benchmark_data_seed(
        function_name=function_name,
        problem_seed=problem_seed,
        rep_index=rep_index,
    )
    payload = ssbp.build_synthetic_sine_benchmark_remote_payload(n, d, function_name, data_seed, 1)
    payload[ssbp.META_KEY]["problem_seed"] = int(problem_seed)
    payload[ssbp.META_KEY]["data_seed"] = int(data_seed)
    payload[ssbp.META_KEY]["rep_index"] = int(rep_index)
    _results_dict(tag)[f"{key}-rep{rep_index}"] = (
        payload,
        n,
        d,
        function_name,
        problem_seed,
        rep_index,
        num_reps,
    )


@app.function(
    image=_modal_image,
    max_containers=20,
    timeout=24 * 60 * 60,
    memory=2 * 1024,
    cpu=1.0,
)
def synthetic_sine_benchmark_batch_resubmitter(batch_of_jobs, tag: str):
    submitted = _submitted_dict(tag)
    todo = []
    for key, job in batch_of_jobs:
        if key in submitted:
            continue
        submitted[key] = True
        todo.append((tag, *job))
    print(f"TODO: {len(todo)}")
    worker = modal.Function.from_name(_get_app_name(tag), "synthetic_sine_benchmark_batch_worker")
    worker.spawn_map(todo)


@app.function(image=_modal_image, max_containers=1, timeout=60 * 60)
def synthetic_sine_benchmark_batch_deleter(keys, tag: str):
    results = _results_dict(tag)
    submitted = _submitted_dict(tag)
    for key in keys:
        try:
            del results[key]
        except KeyError:
            pass
        try:
            del submitted[key]
        except KeyError:
            pass


def _submit_missing(tag: str, jobs_fn: str, output_dir: str | Path, num_reps: int):
    batch = []
    submitted = 0
    submitted_reps: dict[tuple[int, int, str, int, int], list[int]] = {}

    def _flush():
        nonlocal batch
        if not batch:
            return
        func = modal.Function.from_name(_get_app_name(tag), "synthetic_sine_benchmark_batch_resubmitter")
        func.spawn(batch, tag)
        batch = []

    for key, job in _iter_missing_jobs(jobs_fn, output_dir, int(num_reps)):
        batch.append((key, job))
        submitted += 1
        n, d, function_name, problem_seed, rep_index, reps_total = job
        cfg = (n, d, function_name, problem_seed, reps_total)
        submitted_reps.setdefault(cfg, []).append(int(rep_index))
        if len(batch) >= 1000:
            _flush()
    _flush()
    for n, d, function_name, problem_seed, reps_total in sorted(submitted_reps):
        rep_indices = sorted(submitted_reps[(n, d, function_name, problem_seed, reps_total)])
        data_seeds = [
            synthetic_benchmark_data_seed(
                function_name=function_name,
                problem_seed=problem_seed,
                rep_index=rep_index,
            )
            for rep_index in rep_indices
        ]
        print(
            "data_seed range "
            f"N={n} D={d} fn={function_name} pseed={problem_seed} "
            f"reps={rep_indices[0]}-{rep_indices[-1]} ({len(rep_indices)}/{reps_total}) "
            f"seeds={data_seeds[0]}-{data_seeds[-1]}"
        )
    print(f"submitted {submitted} jobs")


def _collect(tag: str, output_dir: str | Path):
    results = _results_dict(tag)
    collected_keys = []
    touched_configs: set[tuple[int, int, str, int, int]] = set()
    for key, payload in results.items():
        if isinstance(payload, dict):
            dest = Path(output_dir) / f"{key}.json"
            if not dest.exists():
                ssbp.write_synthetic_sine_benchmark_json(dest, payload)
                print(f"wrote {dest.resolve()}")
            collected_keys.append(key)
            continue
        (
            rep_payload,
            n,
            d,
            function_name,
            problem_seed,
            rep_index,
            num_reps,
        ) = payload
        dest = _rep_json_dest(
            output_dir,
            n=n,
            d=d,
            function_name=function_name,
            problem_seed=problem_seed,
            rep_index=rep_index,
        )
        if not dest.exists():
            ssbp.write_synthetic_sine_benchmark_json(dest, rep_payload)
            print(f"wrote {dest.resolve()}")
        touched_configs.add((n, d, function_name, problem_seed, num_reps))
        collected_keys.append(key)
    for n, d, function_name, problem_seed, num_reps in sorted(touched_configs):
        agg_dest = _aggregate_reps_to_dest(
            output_dir,
            n=n,
            d=d,
            function_name=function_name,
            problem_seed=problem_seed,
            num_reps=num_reps,
        )
        if agg_dest is not None:
            print(f"aggregated {agg_dest.resolve()}")
    if collected_keys:
        func = modal.Function.from_name(_get_app_name(tag), "synthetic_sine_benchmark_batch_deleter")
        func.spawn(collected_keys, tag)
    print(f"collected {len(collected_keys)} jobs")


def status(tag: str):
    print(f"results_available = {_results_dict(tag).len()}")
    print(f"submitted = {_submitted_dict(tag).len()}")


def clean_up(tag: str):
    for name in (
        f"synthetic_sine_benchmark_results_{tag}",
        f"synthetic_sine_benchmark_submitted_{tag}",
    ):
        try:
            modal.Dict.delete(name)
            print(f"CLEANUP: deleted dict name={name}")
        except Exception as e:
            print(f"CLEANUP: dict delete failed name={name} err={e!r}")


def stop(tag: str):
    clean_up(tag)


@app.local_entrypoint()
def batches(
    tag: str,
    cmd: str,
    jobs_fn: str = "",
    output_dir: str = "results/synthetic_sine_benchmark",
    num_reps: int = 1,
):
    if cmd == "submit":
        if not jobs_fn:
            raise ValueError("jobs_fn is required for submit")
        _submit_missing(tag, jobs_fn, output_dir, int(num_reps))
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
    "_benchmark_json_dest",
    "_get_app_name",
    "_iter_missing_jobs",
    "_job_key",
    "_results_dict",
    "_submitted_dict",
    "app",
    "batches",
    "clean_up",
    "status",
    "stop",
    "synthetic_sine_benchmark_batch_deleter",
    "synthetic_sine_benchmark_batch_resubmitter",
    "synthetic_sine_benchmark_batch_worker",
]
