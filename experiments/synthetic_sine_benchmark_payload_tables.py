from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from analysis.fitting_time.evaluate import (
    SURROGATE_BENCHMARK_ROWS,
    SyntheticSineSurrogateBenchmark,
)
from experiments.synthetic_sine_benchmark_payload_core import (
    read_synthetic_sine_benchmark_json,
    synthetic_surrogate_benchmark_to_wide_row,
)


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


def wide_surrogate_benchmark_row_to_comparison_records(
    row: Any,
) -> list[dict[str, Any]]:
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


def wide_surrogate_benchmark_row_to_long_records(row: Any) -> list[dict[str, Any]]:
    """Turn one wide benchmark row into one tidy row per surrogate."""
    m = _wide_benchmark_row_as_mapping(row)
    meta = {
        "file": m.get("file"),
        "N": m.get("N"),
        "D": m.get("D"),
        "function_name": m.get("function_name"),
        "problem_seed": m.get("problem_seed"),
        "num_reps": m.get("num_reps"),
    }
    out: list[dict[str, Any]] = []
    for prefix, label in SURROGATE_BENCHMARK_ROWS:
        fk = f"{prefix}_fit_seconds_mu"
        if fk in m:
            out.append(
                {
                    **meta,
                    "surrogate": prefix,
                    "surrogate_label": label,
                    "fit_seconds_mu": m[f"{prefix}_fit_seconds_mu"],
                    "fit_seconds_se": m.get(f"{prefix}_fit_seconds_se", 0.0),
                    "normalized_rmse_mu": m[f"{prefix}_normalized_rmse_mu"],
                    "normalized_rmse_se": m.get(f"{prefix}_normalized_rmse_se", 0.0),
                    "log_likelihood_mu": m[f"{prefix}_log_likelihood_mu"],
                    "log_likelihood_se": m.get(f"{prefix}_log_likelihood_se", 0.0),
                }
            )
        else:
            out.append(
                {
                    **meta,
                    "surrogate": prefix,
                    "surrogate_label": label,
                    "fit_seconds_mu": m[f"{prefix}_fit_seconds"],
                    "fit_seconds_se": 0.0,
                    "normalized_rmse_mu": m[f"{prefix}_normalized_rmse"],
                    "normalized_rmse_se": 0.0,
                    "log_likelihood_mu": m[f"{prefix}_log_likelihood"],
                    "log_likelihood_se": 0.0,
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
    skipped = 0
    for path in sorted(root.glob("*.json")):
        try:
            bench, meta = read_synthetic_sine_benchmark_json(path)
        except ValueError:
            skipped += 1
            continue
        benchmarks.append(bench)
        rows.append(
            {
                "file": path.name,
                **meta,
                **synthetic_surrogate_benchmark_to_wide_row(bench),
            }
        )
    if verbose:
        if not rows:
            print("Warning: no *.json files under", root)
        else:
            msg = f"loaded {len(rows)} runs from {root}"
            if skipped:
                msg += f" (skipped {skipped} partial/per-surrogate files)"
            print(msg)
    return rows, benchmarks


def load_synthetic_sine_benchmark_json_dir_long(
    directory: str | Path,
    *,
    verbose: bool = True,
):
    """Load every ``*.json`` under ``directory`` into a tidy dataframe.

    Handles both full-benchmark files (all surrogates) and per-surrogate files.
    """
    import json

    import pandas as pd

    from analysis.fitting_time.evaluate import SURROGATE_BENCHMARK_ROWS

    root = Path(directory)
    long_rows: list[dict[str, Any]] = []
    full_count = 0
    partial_count = 0

    for path in sorted(root.glob("*.json")):
        with path.open() as f:
            data = json.load(f)

        meta = data.get("_meta", {})

        if "triple" in data:
            surrogate_key = meta.get("surrogate_key")
            if not surrogate_key:
                continue
            label_map = {k: v for k, v in SURROGATE_BENCHMARK_ROWS}
            triple = data["triple"]
            long_rows.append(
                {
                    "file": path.name,
                    "N": meta.get("N"),
                    "D": meta.get("D"),
                    "function_name": meta.get("function_name"),
                    "problem_seed": meta.get("problem_seed"),
                    "surrogate": surrogate_key,
                    "surrogate_label": label_map.get(surrogate_key, surrogate_key),
                    "num_reps": 1,
                    "fit_seconds_mu": triple[0],
                    "fit_seconds_se": 0.0,
                    "normalized_rmse_mu": triple[1],
                    "normalized_rmse_se": 0.0,
                    "log_likelihood_mu": triple[2],
                    "log_likelihood_se": 0.0,
                }
            )
            partial_count += 1
        elif "results" in data:
            try:
                bench, _ = read_synthetic_sine_benchmark_json(path)
            except ValueError:
                continue
            row = {
                "file": path.name,
                **meta,
                **synthetic_surrogate_benchmark_to_wide_row(bench),
            }
            long_rows.extend(wide_surrogate_benchmark_row_to_long_records(row))
            full_count += 1
        else:
            try:
                bench, _ = read_synthetic_sine_benchmark_json(path)
            except ValueError:
                continue
            row = {
                "file": path.name,
                **meta,
                **synthetic_surrogate_benchmark_to_wide_row(bench),
            }
            long_rows.extend(wide_surrogate_benchmark_row_to_long_records(row))
            full_count += 1

    if verbose:
        if not long_rows:
            print("Warning: no *.json files under", root)
        else:
            print(f"loaded {full_count} full + {partial_count} per-surrogate runs from {root}")

    df = pd.DataFrame(long_rows)
    if df.empty:
        return df
    return df.sort_values(["N", "D", "function_name", "problem_seed", "surrogate"]).reset_index(drop=True)
