"""Replicate-aware path and aggregation helpers for synthetic benchmark batches."""

from __future__ import annotations

from pathlib import Path

from analysis.fitting_time.evaluate import SURROGATE_BENCHMARK_KEYS
from analysis.fitting_time.evaluate_triples import aggregate_surrogate_replicates
from experiments import synthetic_sine_benchmark_payload as ssbp


def job_key(
    *,
    n: int,
    d: int,
    function_name: str,
    problem_seed: int,
    num_reps: int,
    surrogate_key: str | None = None,
) -> str:
    base = ssbp.synthetic_sine_benchmark_config_slug(
        n=n,
        d=d,
        function_name=function_name,
        problem_seed=problem_seed,
        num_reps=num_reps,
    )
    if surrogate_key is not None:
        return f"{base}_{surrogate_key}"
    return base


def benchmark_json_dest(
    output_dir: str | Path,
    *,
    n: int,
    d: int,
    function_name: str,
    problem_seed: int,
    num_reps: int,
) -> Path:
    return Path(output_dir) / (f"{job_key(n=n, d=d, function_name=function_name, problem_seed=problem_seed, num_reps=num_reps)}.json")


def rep_json_dest(
    output_dir: str | Path,
    *,
    n: int,
    d: int,
    function_name: str,
    problem_seed: int,
    rep_index: int,
) -> Path:
    return (
        Path(output_dir)
        / "_replicates"
        / (f"{ssbp.synthetic_sine_benchmark_rep_slug(n=n, d=d, function_name=function_name, problem_seed=problem_seed, rep_index=rep_index)}.json")
    )


def surrogate_rep_json_dest(
    output_dir: str | Path,
    *,
    n: int,
    d: int,
    function_name: str,
    problem_seed: int,
    rep_index: int,
    surrogate_key: str,
) -> Path:
    """Path for a single-surrogate replicate result."""
    base_slug = ssbp.synthetic_sine_benchmark_rep_slug(
        n=n,
        d=d,
        function_name=function_name,
        problem_seed=problem_seed,
        rep_index=rep_index,
    )
    return Path(output_dir) / "_replicates" / f"{base_slug}_{surrogate_key}.json"


def legacy_single_rep_dest(
    output_dir: str | Path,
    *,
    n: int,
    d: int,
    function_name: str,
    problem_seed: int,
) -> Path:
    return benchmark_json_dest(
        output_dir,
        n=n,
        d=d,
        function_name=function_name,
        problem_seed=problem_seed,
        num_reps=1,
    )


def existing_rep_payload_path(
    output_dir: str | Path,
    *,
    n: int,
    d: int,
    function_name: str,
    problem_seed: int,
    rep_index: int,
) -> Path | None:
    rep_dest = rep_json_dest(
        output_dir,
        n=n,
        d=d,
        function_name=function_name,
        problem_seed=problem_seed,
        rep_index=rep_index,
    )
    if rep_dest.exists():
        return rep_dest
    if int(rep_index) == 0:
        legacy_dest = legacy_single_rep_dest(
            output_dir,
            n=n,
            d=d,
            function_name=function_name,
            problem_seed=problem_seed,
        )
        if legacy_dest.exists():
            return legacy_dest
    return None


def _bench_to_rep_row(bench) -> dict[str, tuple[float, float, float]]:
    out = {}
    for key, result in bench.results.items():
        out[key] = (
            result.fit_seconds.mu,
            result.normalized_rmse.mu,
            result.log_likelihood.mu,
        )
    return out


def aggregate_surrogate_results_to_rep(
    output_dir: str | Path,
    *,
    n: int,
    d: int,
    function_name: str,
    problem_seed: int,
    rep_index: int,
) -> Path | None:
    """Aggregate per-surrogate JSON files into a single replicate JSON.

    Returns the destination path if all surrogates are present, else None.
    """
    import json

    from analysis.fitting_time.evaluate import (
        SURROGATE_BENCHMARK_KEYS,
        BMResult,
        MuSe,
        SyntheticSineSurrogateBenchmark,
        synthetic_benchmark_data_seed,
    )

    results: dict[str, BMResult] = {}
    for surrogate_key in SURROGATE_BENCHMARK_KEYS:
        surr_path = surrogate_rep_json_dest(
            output_dir,
            n=n,
            d=d,
            function_name=function_name,
            problem_seed=problem_seed,
            rep_index=rep_index,
            surrogate_key=surrogate_key,
        )
        if not surr_path.exists():
            return None
        with surr_path.open() as f:
            data = json.load(f)
        triple = data["triple"]
        results[surrogate_key] = BMResult(
            fit_seconds=MuSe(triple[0], 0.0),
            normalized_rmse=MuSe(triple[1], 0.0),
            log_likelihood=MuSe(triple[2], 0.0),
        )

    bench = SyntheticSineSurrogateBenchmark(results=results)
    data_seed = synthetic_benchmark_data_seed(
        function_name=function_name,
        problem_seed=problem_seed,
        rep_index=rep_index,
    )
    payload = ssbp.synthetic_sine_benchmark_result_to_payload(
        bench,
        n=n,
        d=d,
        function_name=function_name,
        problem_seed=problem_seed,
        num_reps=1,
    )
    payload[ssbp.META_KEY]["data_seed"] = int(data_seed)
    payload[ssbp.META_KEY]["rep_index"] = int(rep_index)

    dest = rep_json_dest(
        output_dir,
        n=n,
        d=d,
        function_name=function_name,
        problem_seed=problem_seed,
        rep_index=rep_index,
    )
    ssbp.write_synthetic_sine_benchmark_json(dest, payload)
    return dest


def aggregate_reps_to_dest(
    output_dir: str | Path,
    *,
    n: int,
    d: int,
    function_name: str,
    problem_seed: int,
    num_reps: int,
) -> Path | None:
    rep_paths: list[Path] = []
    for rep_index in range(int(num_reps)):
        path = existing_rep_payload_path(
            output_dir,
            n=n,
            d=d,
            function_name=function_name,
            problem_seed=problem_seed,
            rep_index=rep_index,
        )
        if path is None:
            return None
        rep_paths.append(path)

    rows = []
    for path in rep_paths:
        bench, _ = ssbp.read_synthetic_sine_benchmark_json(path)
        rows.append(_bench_to_rep_row(bench))
    agg = aggregate_surrogate_replicates(rows)
    payload = ssbp.synthetic_sine_benchmark_result_to_payload(
        agg,
        n=n,
        d=d,
        function_name=function_name,
        problem_seed=problem_seed,
        num_reps=num_reps,
    )
    dest = benchmark_json_dest(
        output_dir,
        n=n,
        d=d,
        function_name=function_name,
        problem_seed=problem_seed,
        num_reps=num_reps,
    )
    ssbp.write_synthetic_sine_benchmark_json(dest, payload)
    return dest


def iter_missing_jobs(jobs_fn: str, output_dir: str | Path, num_reps: int):
    for n, d, function_name, problem_seed in ssbp.load_synthetic_sine_benchmark_jobs(jobs_fn):
        final_dest = benchmark_json_dest(
            output_dir,
            n=n,
            d=d,
            function_name=function_name,
            problem_seed=problem_seed,
            num_reps=num_reps,
        )
        if final_dest.exists():
            print(f"skip existing aggregate {final_dest.resolve()}")
            continue
        for rep_index in range(int(num_reps)):
            existing = existing_rep_payload_path(
                output_dir,
                n=n,
                d=d,
                function_name=function_name,
                problem_seed=problem_seed,
                rep_index=rep_index,
            )
            if existing is not None:
                print(f"skip existing rep {existing.resolve()}")
                continue
            key = job_key(
                n=n,
                d=d,
                function_name=function_name,
                problem_seed=problem_seed,
                num_reps=num_reps,
            )
            yield (
                f"{key}-rep{rep_index}",
                (n, d, function_name, problem_seed, rep_index, num_reps),
            )


def iter_missing_surrogate_jobs(jobs_fn: str, output_dir: str | Path, num_reps: int):
    """Iterate over missing (config, rep, surrogate) jobs.

    Each job handles a single surrogate for a single replicate, yielding:
        (unique_key, (n, d, function_name, problem_seed, rep_index, num_reps, surrogate_key))
    """
    for n, d, function_name, problem_seed in ssbp.load_synthetic_sine_benchmark_jobs(jobs_fn):
        for rep_index in range(int(num_reps)):
            for surrogate_key in SURROGATE_BENCHMARK_KEYS:
                surr_dest = surrogate_rep_json_dest(
                    output_dir,
                    n=n,
                    d=d,
                    function_name=function_name,
                    problem_seed=problem_seed,
                    rep_index=rep_index,
                    surrogate_key=surrogate_key,
                )
                if surr_dest.exists():
                    continue
                key = job_key(
                    n=n,
                    d=d,
                    function_name=function_name,
                    problem_seed=problem_seed,
                    num_reps=num_reps,
                    surrogate_key=surrogate_key,
                )
                yield (
                    f"{key}-rep{rep_index}",
                    (
                        n,
                        d,
                        function_name,
                        problem_seed,
                        rep_index,
                        num_reps,
                        surrogate_key,
                    ),
                )
