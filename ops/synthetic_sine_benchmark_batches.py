#!/usr/bin/env python3
"""Deploy/submit/collect/stop wrapper for synthetic surrogate benchmark batches."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import click


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_repo_imports() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _resolve_surrogate_key(surrogate: str) -> str:
    """Map CLI surrogate name to :data:`SURROGATE_BENCHMARK_KEYS` entry."""
    _ensure_repo_imports()
    from analysis.fitting_time.evaluate_metrics import (
        SURROGATE_BENCHMARK_KEYS,
        SURROGATE_BENCHMARK_ROWS,
    )

    s = surrogate.strip()
    if not s:
        raise click.BadParameter("surrogate must be non-empty")
    key_like = "_".join(s.lower().replace("-", "_").split())
    if key_like in SURROGATE_BENCHMARK_KEYS:
        return key_like
    aliases = {"smac": "smac_rf", "smacrf": "smac_rf"}
    if key_like in aliases:
        return aliases[key_like]
    compact = key_like.replace("_", "")
    for key, label in SURROGATE_BENCHMARK_ROWS:
        label_cmp = label.lower().replace(" ", "").replace("_", "")
        if compact == label_cmp:
            return key
    raise click.BadParameter(f"unknown surrogate {surrogate!r}; expected one of {', '.join(SURROGATE_BENCHMARK_KEYS)}")


def _get_impl_path() -> str:
    return "experiments/modal_synthetic_sine_benchmark_batches_impl.py"


def _get_app_name(tag: str) -> str:
    return f"yubo-synth-sine-batch-{tag}"


def _run_modal(args: list[str], tag: str) -> None:
    env = os.environ.copy()
    env["MODAL_TAG"] = tag
    cmd = ["modal", *args]
    click.echo(f"Running: {' '.join(cmd)} (MODAL_TAG={tag})")
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("tag")
def deploy(tag: str):
    """Deploy the batch Modal app for this tag."""
    _run_modal(["deploy", _get_impl_path()], tag)


@cli.command()
@click.argument("tag")
@click.argument("jobs_fn")
@click.option(
    "--output-dir",
    default="results/synthetic_sine_benchmark",
    show_default=True,
)
@click.option("--num-reps", default=1, type=int, show_default=True)
def submit(tag: str, jobs_fn: str, output_dir: str, num_reps: int):
    """Submit jobs that are not already present on disk."""
    impl = _get_impl_path()
    _run_modal(
        [
            "run",
            f"{impl}::batches",
            "--tag",
            tag,
            "--cmd",
            "submit",
            "--jobs-fn",
            jobs_fn,
            "--output-dir",
            output_dir,
            "--num-reps",
            str(num_reps),
        ],
        tag,
    )


@cli.command()
@click.argument("tag")
@click.option(
    "--output-dir",
    default="results/synthetic_sine_benchmark",
    show_default=True,
)
def collect(tag: str, output_dir: str):
    """Collect completed results to the local output directory."""
    impl = _get_impl_path()
    _run_modal(
        [
            "run",
            f"{impl}::batches",
            "--tag",
            tag,
            "--cmd",
            "collect",
            "--output-dir",
            output_dir,
        ],
        tag,
    )


@cli.command()
@click.argument("tag")
def status(tag: str):
    """Show submitted/result dict sizes for this tag."""
    impl = _get_impl_path()
    _run_modal(["run", f"{impl}::batches", "--tag", tag, "--cmd", "status"], tag)


@cli.command()
@click.argument("tag")
def stop(tag: str):
    """Stop the app and clean up the backing Modal dicts."""
    app_name = _get_app_name(tag)
    click.echo(f"Stopping app: {app_name}")
    stop_result = subprocess.run(["modal", "app", "stop", app_name])
    if stop_result.returncode != 0:
        click.echo(f"Warning: modal app stop returned {stop_result.returncode}")

    impl = _get_impl_path()
    env = os.environ.copy()
    env["MODAL_TAG"] = tag
    click.echo(f"Cleaning up dicts for tag: {tag}")
    subprocess.run(["modal", "run", f"{impl}::batches", "--tag", tag, "--cmd", "stop"], env=env)


_LOCAL_SINGLE_PARAMS = [
    click.Argument(["n_train"], type=int, required=True),
    click.Argument(["function_name"], required=True),
    click.Argument(["rep_index"], type=int, required=True),
    click.Argument(["surrogate"], required=True),
    click.Option(
        ("-d", "--d", "d_dims"),
        type=int,
        default=10,
        show_default=True,
        help="Dimensionality.",
    ),
    click.Option(
        ("--problem-seed", "problem_seed"),
        type=int,
        default=17,
        show_default=True,
    ),
    click.Option(
        ("--output-dir", "output_dir"),
        type=click.Path(),
        default="results/synthetic_sine_benchmark",
        show_default=True,
    ),
    click.Option(
        ("--num-reps", "num_reps"),
        type=int,
        default=10,
        show_default=True,
        help="With --aggregate: replicate count passed to rep- and config-level rollup helpers.",
    ),
    click.Option(
        ("--aggregate/--no-aggregate", "aggregate"),
        default=False,
        help="Optional offline rollup to combined rep / nrep JSON (not part of the Modal batch pipeline).",
    ),
]


@cli.command(
    "local-single",
    params=_LOCAL_SINGLE_PARAMS,
    short_help="N TARGET REP_INDEX SURROGATE [options]",
)
def local_single(
    n_train: int,
    function_name: str,
    rep_index: int,
    surrogate: str,
    d_dims: int,
    problem_seed: int,
    output_dir: str,
    num_reps: int,
    aggregate: bool,
) -> None:
    """Run one surrogate × one rep locally if the per-surrogate JSON is missing.

    Arguments: N, TARGET (e.g. sphere, ackley), REP_INDEX (0-based), SURROGATE (key or label).
    Use -d/--d to override dimensionality (default 10).
    """
    if n_train < 1 or d_dims < 1:
        raise click.BadParameter("N and D must be positive")
    if rep_index < 0:
        raise click.BadParameter("REP_INDEX must be >= 0")
    if num_reps < 1:
        raise click.BadParameter("num-reps must be >= 1")

    _ensure_repo_imports()
    from analysis.fitting_time.evaluate import (
        benchmark_single_surrogate_with_data,
        normalize_benchmark_function_name,
        synthetic_benchmark_data_seed,
    )
    from experiments import synthetic_sine_benchmark_payload as ssbp
    from experiments.modal_synthetic_sine_benchmark_batches_reps import (
        aggregate_reps_to_dest,
        aggregate_surrogate_results_to_rep,
        surrogate_rep_json_dest,
    )

    surrogate_key = _resolve_surrogate_key(surrogate)
    fn = normalize_benchmark_function_name(function_name)
    out = Path(output_dir)
    dest = surrogate_rep_json_dest(
        out,
        n=n_train,
        d=d_dims,
        function_name=fn,
        problem_seed=problem_seed,
        rep_index=rep_index,
        surrogate_key=surrogate_key,
    )
    if dest.exists():
        click.echo(f"skip existing {dest.resolve()}")
        return

    data_seed = synthetic_benchmark_data_seed(function_name=fn, problem_seed=problem_seed, rep_index=rep_index)
    click.echo(
        f"running surrogate={surrogate_key} N={n_train} D={d_dims} fn={fn} problem_seed={problem_seed} rep_index={rep_index} data_seed={data_seed}",
        err=True,
    )
    triple = benchmark_single_surrogate_with_data(
        N=n_train,
        D=d_dims,
        function_name=fn,
        surrogate_key=surrogate_key,
        data_seed=data_seed,
    )
    payload = {
        "triple": list(triple),
        "_meta": {
            "N": int(n_train),
            "D": int(d_dims),
            "function_name": fn,
            "problem_seed": int(problem_seed),
            "data_seed": int(data_seed),
            "rep_index": int(rep_index),
            "surrogate_key": surrogate_key,
        },
    }
    ssbp.write_synthetic_sine_benchmark_json(dest, payload)
    click.echo(f"wrote {dest.resolve()}")

    if aggregate:
        rep_dest = aggregate_surrogate_results_to_rep(
            out,
            n=n_train,
            d=d_dims,
            function_name=fn,
            problem_seed=problem_seed,
            rep_index=rep_index,
        )
        if rep_dest is not None:
            click.echo(f"aggregated surrogates -> {rep_dest.resolve()}")
        cfg_dest = aggregate_reps_to_dest(
            out,
            n=n_train,
            d=d_dims,
            function_name=fn,
            problem_seed=problem_seed,
            num_reps=num_reps,
        )
        if cfg_dest is not None:
            click.echo(f"aggregated reps -> {cfg_dest.resolve()}")


if __name__ == "__main__":
    cli()
