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
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import modal

from analysis.fitting_time.evaluate import (
    SURROGATE_BENCHMARK_KEYS,
    SURROGATE_BENCHMARK_ROWS,
    BMResult,
    MuSe,
    SyntheticSineSurrogateBenchmark,
    benchmark_synthetic_sine_surrogates,
    normalize_benchmark_function_name,
)

META_KEY = "_meta"

__all__ = [
    "META_KEY",
    "build_synthetic_sine_benchmark_remote_payload",
    "load_synthetic_sine_benchmark_jobs",
    "load_synthetic_sine_benchmark_json_dir",
    "read_synthetic_sine_benchmark_json",
    "run_synthetic_sine_benchmark_modal_to_disk",
    "synthetic_surrogate_benchmark_row_caption",
    "synthetic_surrogate_benchmark_to_wide_row",
    "synthetic_sine_benchmark_config_slug",
    "synthetic_sine_benchmark_from_payload",
    "synthetic_sine_benchmark_result_to_payload",
    "wide_surrogate_benchmark_row_to_comparison_records",
    "write_synthetic_sine_benchmark_json",
]

_LEGACY_TRIPLE_KEYS: frozenset[str] = frozenset(
    f"{prefix}_{suffix}" for prefix in SURROGATE_BENCHMARK_KEYS for suffix in ("fit_seconds", "normalized_rmse", "log_likelihood")
)


def _legacy_flat_payload_to_bench(d: Mapping[str, Any]) -> SyntheticSineSurrogateBenchmark:
    results: dict[str, BMResult] = {}
    for prefix in SURROGATE_BENCHMARK_KEYS:
        results[prefix] = BMResult(
            MuSe(float(d[f"{prefix}_fit_seconds"]), 0.0),
            MuSe(float(d[f"{prefix}_normalized_rmse"]), 0.0),
            MuSe(float(d[f"{prefix}_log_likelihood"]), 0.0),
        )
    return SyntheticSineSurrogateBenchmark(results=results)


def _bench_from_nested_results(obj: Any) -> SyntheticSineSurrogateBenchmark:
    if not isinstance(obj, dict):
        raise TypeError("results must be a dict")
    results: dict[str, BMResult] = {}
    for prefix in SURROGATE_BENCHMARK_KEYS:
        block = obj[prefix]
        results[prefix] = BMResult(
            fit_seconds=MuSe(float(block["fit_seconds"]["mu"]), float(block["fit_seconds"]["se"])),
            normalized_rmse=MuSe(float(block["normalized_rmse"]["mu"]), float(block["normalized_rmse"]["se"])),
            log_likelihood=MuSe(float(block["log_likelihood"]["mu"]), float(block["log_likelihood"]["se"])),
        )
    return SyntheticSineSurrogateBenchmark(results=results)


def synthetic_surrogate_benchmark_to_wide_row(bench: SyntheticSineSurrogateBenchmark) -> dict[str, Any]:
    """Flatten :class:`~analysis.fitting_time.evaluate.SyntheticSineSurrogateBenchmark` for tabular rows."""
    flat: dict[str, Any] = {}
    for prefix in SURROGATE_BENCHMARK_KEYS:
        br = bench.results[prefix]
        flat[f"{prefix}_fit_seconds_mu"] = br.fit_seconds.mu
        flat[f"{prefix}_fit_seconds_se"] = br.fit_seconds.se
        flat[f"{prefix}_normalized_rmse_mu"] = br.normalized_rmse.mu
        flat[f"{prefix}_normalized_rmse_se"] = br.normalized_rmse.se
        flat[f"{prefix}_log_likelihood_mu"] = br.log_likelihood.mu
        flat[f"{prefix}_log_likelihood_se"] = br.log_likelihood.se
    return flat


def synthetic_sine_benchmark_result_to_payload(
    result: SyntheticSineSurrogateBenchmark,
    *,
    n: int,
    d: int,
    function_name: str,
    problem_seed: int,
    num_reps: int = 1,
) -> dict:
    """Merge :func:`dataclasses.asdict` with run metadata for JSON persistence."""
    out = asdict(result)
    fn = normalize_benchmark_function_name(function_name)
    out[META_KEY] = {
        "N": n,
        "D": d,
        "function_name": fn,
        "problem_seed": problem_seed,
        "num_reps": int(num_reps),
    }
    return out


def synthetic_sine_benchmark_from_payload(data: dict) -> tuple[SyntheticSineSurrogateBenchmark, dict]:
    """Load a benchmark dataclass and metadata dict written by :func:`synthetic_sine_benchmark_result_to_payload`."""
    d = dict(data)
    meta = dict(d.pop(META_KEY, {}))
    if "results" in d:
        nested = d.pop("results")
        bench = _bench_from_nested_results(nested)
        return bench, meta
    if _LEGACY_TRIPLE_KEYS.intersection(d.keys()):
        bench = _legacy_flat_payload_to_bench(d)
        for k in _LEGACY_TRIPLE_KEYS:
            d.pop(k, None)
        return bench, meta
    raise ValueError("payload missing surrogate results (expected 'results' or legacy *_{fit_seconds,...} keys)")


def synthetic_sine_benchmark_config_slug(*, n: int, d: int, function_name: str, problem_seed: int, num_reps: int = 1) -> str:
    """Filesystem-safe filename stem for one config (``.json`` appended by caller)."""
    fn = normalize_benchmark_function_name(function_name)
    fn_part = re.sub(r"[^a-zA-Z0-9._-]+", "_", fn).strip("_") or "fn"
    stem = f"N{n}_D{d}_{fn_part}_pseed{problem_seed}"
    if int(num_reps) != 1:
        stem += f"_nrep{int(num_reps)}"
    return stem


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


def _wide_benchmark_row_as_mapping(row: Any) -> dict[str, Any]:
    if hasattr(row, "iloc") and hasattr(row, "shape") and hasattr(row, "columns"):
        n = int(row.shape[0])
        if n != 1:
            raise ValueError(f"expected exactly one DataFrame row, got {n}")
        row = row.iloc[0]
    if hasattr(row, "to_dict") and callable(row.to_dict):
        return dict(row.to_dict())
    if isinstance(row, Mapping):
        return dict(row)
    raise TypeError("row must be a one-row DataFrame, a Series, or a dict-like mapping")


def wide_surrogate_benchmark_row_to_comparison_records(row: Any) -> list[dict[str, Any]]:
    """Turn one wide benchmark row into tidy rows for a human-readable table.

    ``row`` is typically ``df.loc[i]`` or ``df[mask].iloc[0]`` from
    :func:`load_synthetic_sine_benchmark_json_dir` output.  Each returned dict has
    surrogate label plus ``μ`` / ``SE`` columns for fit time, NRMSE, and log-likelihood.
    Legacy rows (single ``*_fit_seconds`` float per surrogate) are still accepted.
    """
    m = _wide_benchmark_row_as_mapping(row)
    out: list[dict[str, Any]] = []
    for prefix, label in SURROGATE_BENCHMARK_ROWS:
        fk = f"{prefix}_fit_seconds_mu"
        if fk in m:
            out.append(
                {
                    "Surrogate": label,
                    "Fit (s) μ": m[fk],
                    "Fit (s) SE": m.get(f"{prefix}_fit_seconds_se", 0.0),
                    "NRMSE μ": m[f"{prefix}_normalized_rmse_mu"],
                    "NRMSE SE": m.get(f"{prefix}_normalized_rmse_se", 0.0),
                    "LogLik (nats) μ": m[f"{prefix}_log_likelihood_mu"],
                    "LogLik (nats) SE": m.get(f"{prefix}_log_likelihood_se", 0.0),
                }
            )
        else:
            out.append(
                {
                    "Surrogate": label,
                    "Fit (s) μ": m[f"{prefix}_fit_seconds"],
                    "Fit (s) SE": 0.0,
                    "NRMSE μ": m[f"{prefix}_normalized_rmse"],
                    "NRMSE SE": 0.0,
                    "LogLik (nats) μ": m[f"{prefix}_log_likelihood"],
                    "LogLik (nats) SE": 0.0,
                }
            )
    return out


def synthetic_surrogate_benchmark_row_caption(row: Any) -> str:
    """One-line description: ``N``, ``D``, target, ``problem_seed``, optional ``file``."""
    m = _wide_benchmark_row_as_mapping(row)
    parts = [
        f"N={m.get('N')}",
        f"D={m.get('D')}",
        f"target={m.get('function_name')}",
        f"problem_seed={m.get('problem_seed')}",
    ]
    if m.get("file"):
        parts.append(str(m["file"]))
    return ", ".join(parts)


def load_synthetic_sine_benchmark_json_dir(
    directory: str | Path,
    *,
    verbose: bool = True,
) -> tuple[list[dict], list[SyntheticSineSurrogateBenchmark]]:
    """Load every ``*.json`` under ``directory`` (sorted by filename).

    Each row dict has ``file`` (basename), benchmark metadata (``N``, ``D``,
    ``function_name``, ``problem_seed``, ``num_reps``), and flattened surrogate
    ``*_mu`` / ``*_se`` columns from :func:`synthetic_surrogate_benchmark_to_wide_row`.
    """
    root = Path(directory)
    rows: list[dict] = []
    benchmarks: list[SyntheticSineSurrogateBenchmark] = []
    for path in sorted(root.glob("*.json")):
        bench, meta = read_synthetic_sine_benchmark_json(path)
        benchmarks.append(bench)
        rows.append({"file": path.name, **meta, **synthetic_surrogate_benchmark_to_wide_row(bench)})
    if verbose:
        if not rows:
            print("Warning: no *.json files under", root)
        else:
            print(f"loaded {len(rows)} runs from {root}")
    return rows, benchmarks


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


def build_synthetic_sine_benchmark_remote_payload(n: int, d: int, function_name: str, problem_seed: int, num_reps: int = 1) -> dict:
    """Run the benchmark and return JSON-serializable payload (used by the Modal worker)."""
    fn = normalize_benchmark_function_name(function_name)
    r = benchmark_synthetic_sine_surrogates(N=n, D=d, function_name=fn, problem_seed=problem_seed, num_reps=int(num_reps))
    return synthetic_sine_benchmark_result_to_payload(r, n=n, d=d, function_name=fn, problem_seed=problem_seed, num_reps=int(num_reps))


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
    num_reps: int = 1,
) -> Path:
    """When ``start_app`` is false, skip ``app.run()`` (Modal ``local_entrypoint`` already runs the app)."""
    fn = normalize_benchmark_function_name(function_name)
    out_root = Path(output_dir)
    slug = synthetic_sine_benchmark_config_slug(n=n, d=d, function_name=fn, problem_seed=problem_seed, num_reps=int(num_reps))
    dest = out_root / f"{slug}.json"
    nr = int(num_reps)
    with modal.enable_output():
        if start_app:
            with app.run():
                payload = remote_fn.remote(n, d, fn, int(problem_seed), nr)
        else:
            payload = remote_fn.remote(n, d, fn, int(problem_seed), nr)
    write_synthetic_sine_benchmark_json(dest, payload)
    return dest
