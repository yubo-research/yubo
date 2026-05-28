"""JSON/slug helpers for surrogate timing benchmarks (no Modal dependency).

Each run is tagged by a **required** ``function_name`` (e.g. ``sine`` for the FittingTime
target, or ``sphere`` / ``ackley`` for :mod:`problems.pure_functions`). No Modal image
build: safe to import in tests without ``mk_image()`` cost.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from analysis.fitting_time.evaluate import (
    SURROGATE_BENCHMARK_KEYS,
    BMResult,
    MuSe,
    SyntheticSineSurrogateBenchmark,
    normalize_benchmark_function_name,
)

META_KEY = "_meta"

_LEGACY_TRIPLE_KEYS: frozenset[str] = frozenset(
    f"{prefix}_{suffix}" for prefix in SURROGATE_BENCHMARK_KEYS for suffix in ("fit_seconds", "normalized_rmse", "log_likelihood")
)


def _legacy_flat_payload_to_bench(
    d: Mapping[str, Any],
) -> SyntheticSineSurrogateBenchmark:
    results: dict[str, BMResult] = {}
    for prefix in SURROGATE_BENCHMARK_KEYS:
        fit_key = f"{prefix}_fit_seconds"
        if fit_key not in d:
            results[prefix] = BMResult(
                MuSe(math.nan, math.nan),
                MuSe(math.nan, math.nan),
                MuSe(math.nan, math.nan),
            )
            continue
        results[prefix] = BMResult(
            MuSe(float(d[fit_key]), 0.0),
            MuSe(float(d[f"{prefix}_normalized_rmse"]), 0.0),
            MuSe(float(d[f"{prefix}_log_likelihood"]), 0.0),
        )
    return SyntheticSineSurrogateBenchmark(results=results)


def _bench_from_nested_results(obj: Any) -> SyntheticSineSurrogateBenchmark:
    if not isinstance(obj, dict):
        raise TypeError("results must be a dict")
    results: dict[str, BMResult] = {}
    for prefix in SURROGATE_BENCHMARK_KEYS:
        if prefix not in obj:
            results[prefix] = BMResult(
                MuSe(math.nan, math.nan),
                MuSe(math.nan, math.nan),
                MuSe(math.nan, math.nan),
            )
            continue
        block = obj[prefix]
        results[prefix] = BMResult(
            fit_seconds=MuSe(float(block["fit_seconds"]["mu"]), float(block["fit_seconds"]["se"])),
            normalized_rmse=MuSe(
                float(block["normalized_rmse"]["mu"]),
                float(block["normalized_rmse"]["se"]),
            ),
            log_likelihood=MuSe(
                float(block["log_likelihood"]["mu"]),
                float(block["log_likelihood"]["se"]),
            ),
        )
    return SyntheticSineSurrogateBenchmark(results=results)


def synthetic_surrogate_benchmark_to_wide_row(
    bench: SyntheticSineSurrogateBenchmark,
) -> dict[str, Any]:
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


def synthetic_sine_benchmark_from_payload(
    data: dict,
) -> tuple[SyntheticSineSurrogateBenchmark, dict]:
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


def synthetic_sine_benchmark_rep_slug(
    *,
    n: int,
    d: int,
    function_name: str,
    problem_seed: int,
    rep_index: int,
) -> str:
    """Filesystem-safe filename stem for one replicate payload."""
    if int(rep_index) < 0:
        raise ValueError("rep_index must be >= 0")
    base = synthetic_sine_benchmark_config_slug(
        n=n,
        d=d,
        function_name=function_name,
        problem_seed=problem_seed,
        num_reps=1,
    )
    return f"{base}_rep{int(rep_index)}"


def write_synthetic_sine_benchmark_json(path: Path, payload: dict) -> None:
    """Write JSON with NaN/inf preserved (Python ``json`` non-standard tokens).

    Strict JSON parsers reject ``NaN``/``Infinity``; use Python ``json.load`` or a
    tolerant consumer when reading these files.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)


def read_synthetic_sine_benchmark_json(
    path: Path,
) -> tuple[SyntheticSineSurrogateBenchmark, dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return synthetic_sine_benchmark_from_payload(data)
