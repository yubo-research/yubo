#!/usr/bin/env python3
"""Deploy/submit/collect/stop wrapper for synthetic surrogate benchmark batches."""

from __future__ import annotations

import os
import subprocess
import sys

import click


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


if __name__ == "__main__":
    cli()
