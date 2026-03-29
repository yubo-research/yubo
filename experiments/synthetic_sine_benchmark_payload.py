"""JSON/slug helpers and remote payload builder for synthetic sine surrogate benchmarks.

No Modal image build: safe to import in tests without ``mk_image()`` cost.
"""

from __future__ import annotations

import json
import re
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
    function_name: str | None,
    problem_seed: int,
) -> dict:
    """Merge :func:`dataclasses.asdict` with run metadata for JSON persistence."""
    out = asdict(result)
    out[META_KEY] = {"N": n, "D": d, "function_name": function_name, "problem_seed": problem_seed}
    return out


def synthetic_sine_benchmark_from_payload(data: dict) -> tuple[SyntheticSineSurrogateBenchmark, dict]:
    """Load a benchmark dataclass and metadata dict written by :func:`synthetic_sine_benchmark_result_to_payload`."""
    d = dict(data)
    meta = d.pop(META_KEY, {})
    names = {f.name for f in fields(SyntheticSineSurrogateBenchmark)}
    bench_kw = {k: v for k, v in d.items() if k in names}
    return SyntheticSineSurrogateBenchmark(**bench_kw), meta


def synthetic_sine_benchmark_config_slug(*, n: int, d: int, function_name: str | None, problem_seed: int) -> str:
    """Filesystem-safe filename stem for one config (``.json`` appended by caller)."""
    if function_name is None:
        fn_part = "sine"
    else:
        fn_part = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(function_name)).strip("_") or "fn"
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


def build_synthetic_sine_benchmark_remote_payload(n: int, d: int, function_name: str | None, problem_seed: int) -> dict:
    """Run the benchmark and return JSON-serializable payload (used by the Modal worker)."""
    fn = normalize_benchmark_function_name(function_name)
    r = benchmark_synthetic_sine_surrogates(N=n, D=d, function_name=fn, problem_seed=problem_seed)
    return synthetic_sine_benchmark_result_to_payload(r, n=n, d=d, function_name=fn, problem_seed=problem_seed)


def run_synthetic_sine_benchmark_modal_to_disk(
    n: int,
    d: int,
    function_name: str | None,
    problem_seed: int,
    output_dir: str | Path,
    *,
    app: Any,
    remote_fn: Any,
) -> Path:
    if function_name is None:
        fn = None
    else:
        stripped = str(function_name).strip()
        fn = None if not stripped else stripped
    out_root = Path(output_dir)
    slug = synthetic_sine_benchmark_config_slug(n=n, d=d, function_name=fn, problem_seed=problem_seed)
    dest = out_root / f"{slug}.json"
    with modal.enable_output():
        with app.run():
            payload = remote_fn.remote(n, d, fn, int(problem_seed))
    write_synthetic_sine_benchmark_json(dest, payload)
    return dest
