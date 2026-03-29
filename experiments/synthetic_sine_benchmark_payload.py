"""JSON/slug helpers and remote payload builder for surrogate timing benchmarks.

Each run is tagged by a **required** ``function_name`` (e.g. ``sine`` for the FittingTime
target, or ``sphere`` / ``ackley`` for :mod:`problems.pure_functions`). No Modal image
build: safe to import in tests without ``mk_image()`` cost.
"""

from __future__ import annotations

import importlib
import json
import re
import types
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import modal

from analysis.fitting_time.evaluate import (
    SyntheticSineSurrogateBenchmark,
    benchmark_synthetic_sine_surrogates,
    normalize_benchmark_function_name,
)

META_KEY = "_meta"

__all__ = [
    "META_KEY",
    "build_synthetic_sine_benchmark_remote_payload",
    "load_synthetic_sine_benchmark_jobs",
    "read_synthetic_sine_benchmark_json",
    "run_synthetic_sine_benchmark_modal_to_disk",
    "synthetic_sine_benchmark_config_slug",
    "synthetic_sine_benchmark_from_payload",
    "synthetic_sine_benchmark_result_to_payload",
    "write_synthetic_sine_benchmark_json",
]


def synthetic_sine_benchmark_result_to_payload(
    result: SyntheticSineSurrogateBenchmark,
    *,
    n: int,
    d: int,
    function_name: str,
    problem_seed: int,
) -> dict:
    """Merge :func:`dataclasses.asdict` with run metadata for JSON persistence."""
    out = asdict(result)
    fn = normalize_benchmark_function_name(function_name)
    out[META_KEY] = {"N": n, "D": d, "function_name": fn, "problem_seed": problem_seed}
    return out


def synthetic_sine_benchmark_from_payload(data: dict) -> tuple[SyntheticSineSurrogateBenchmark, dict]:
    """Load a benchmark dataclass and metadata dict written by :func:`synthetic_sine_benchmark_result_to_payload`."""
    d = dict(data)
    meta = d.pop(META_KEY, {})
    names = {f.name for f in fields(SyntheticSineSurrogateBenchmark)}
    bench_kw = {k: v for k, v in d.items() if k in names}
    return SyntheticSineSurrogateBenchmark(**bench_kw), meta


def synthetic_sine_benchmark_config_slug(*, n: int, d: int, function_name: str, problem_seed: int) -> str:
    """Filesystem-safe filename stem for one config (``.json`` appended by caller)."""
    fn = normalize_benchmark_function_name(function_name)
    fn_part = re.sub(r"[^a-zA-Z0-9._-]+", "_", fn).strip("_") or "fn"
    return f"N{n}_D{d}_{fn_part}_pseed{problem_seed}"


def write_synthetic_sine_benchmark_json(path: Path, payload: dict) -> None:
    """Write JSON with NaN/inf preserved (Python ``json`` non-standard tokens).

    Strict JSON parsers reject ``NaN``/``Infinity``; use Python ``json.load`` or a
    tolerant consumer when reading these files.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)


def read_synthetic_sine_benchmark_json(path: Path) -> tuple[SyntheticSineSurrogateBenchmark, dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return synthetic_sine_benchmark_from_payload(data)


def load_synthetic_sine_benchmark_jobs(
    jobs_fn: str,
    *,
    _batch_jobs_module: types.ModuleType | None = None,
) -> list[tuple[int, int, str, int]]:
    """Call ``jobs_fn`` on :mod:`analysis.fitting_time.batch_jobs`; return normalized rows.

    ``jobs_fn`` must be a single Python identifier naming a **parameterless** callable that
    returns a non-empty ``list`` of :class:`~analysis.fitting_time.batch_jobs.SyntheticBenchJob`.
    The ``_batch_jobs_module`` argument is for tests only (inject a fake module).
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", jobs_fn):
        raise ValueError("jobs_fn must be a single Python identifier (e.g. example_sphere_n12_d2_seed0)")
    mod = _batch_jobs_module
    if mod is None:
        mod = importlib.import_module("analysis.fitting_time.batch_jobs")
    job_callable = getattr(mod, jobs_fn, None)
    if job_callable is None or not callable(job_callable):
        raise ValueError(f"analysis.fitting_time.batch_jobs.{jobs_fn} is missing or not callable")
    job_cls = getattr(mod, "SyntheticBenchJob", None)
    if job_cls is None:
        raise ValueError("batch_jobs module must define SyntheticBenchJob")

    raw = job_callable()
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(f"{jobs_fn}() must return a non-empty list")
    out: list[tuple[int, int, str, int]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, job_cls):
            raise TypeError(f"{jobs_fn}()[{i}] must be SyntheticBenchJob, got {type(item).__name__}")
        if item.n < 1 or item.d < 1:
            raise ValueError(f"{jobs_fn}()[{i}]: n and d must be positive, got n={item.n}, d={item.d}")
        fn = normalize_benchmark_function_name(item.target)
        out.append((item.n, item.d, fn, int(item.problem_seed)))
    return out


def build_synthetic_sine_benchmark_remote_payload(n: int, d: int, function_name: str, problem_seed: int) -> dict:
    """Run the benchmark and return JSON-serializable payload (used by the Modal worker)."""
    fn = normalize_benchmark_function_name(function_name)
    r = benchmark_synthetic_sine_surrogates(N=n, D=d, function_name=fn, problem_seed=problem_seed)
    return synthetic_sine_benchmark_result_to_payload(r, n=n, d=d, function_name=fn, problem_seed=problem_seed)


def run_synthetic_sine_benchmark_modal_to_disk(
    n: int,
    d: int,
    function_name: str,
    problem_seed: int,
    output_dir: str | Path,
    *,
    app: Any,
    remote_fn: Any,
    start_app: bool = True,
) -> Path:
    """When ``start_app`` is false, skip ``app.run()`` (Modal ``local_entrypoint`` already runs the app)."""
    fn = normalize_benchmark_function_name(function_name)
    out_root = Path(output_dir)
    slug = synthetic_sine_benchmark_config_slug(n=n, d=d, function_name=fn, problem_seed=problem_seed)
    dest = out_root / f"{slug}.json"
    with modal.enable_output():
        if start_app:
            with app.run():
                payload = remote_fn.remote(n, d, fn, int(problem_seed))
        else:
            payload = remote_fn.remote(n, d, fn, int(problem_seed))
    write_synthetic_sine_benchmark_json(dest, payload)
    return dest
