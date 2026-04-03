#!/usr/bin/env python3
import os
import subprocess
import sys

import click


def _get_impl_path() -> str:
    """Return path to modal_batches_impl.py relative to repo root."""
    return "experiments/modal_batches_impl.py"


def _run_modal(args: list[str], tag: str) -> None:
    """Run modal command with MODAL_TAG set."""
    env = os.environ.copy()
    env["MODAL_TAG"] = tag
    cmd = ["modal"] + args
    click.echo(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("tag")
def deploy(tag: str):
    """Deploy the modal app with the given tag."""
    _run_modal(["deploy", _get_impl_path()], tag)


@cli.command()
@click.argument("tag")
@click.argument("batch_tag")
def submit(tag: str, batch_tag: str):
    """Submit missing jobs for the batch."""
    impl = _get_impl_path()
    _run_modal(["run", f"{impl}::batches", "--tag", tag, "--cmd", "submit-missing", "--batch-tag", batch_tag], tag)


@cli.command("submit-force")
@click.argument("tag")
@click.argument("batch_tag")
def submit_force(tag: str, batch_tag: str):
    """Force resubmit all jobs for the batch."""
    impl = _get_impl_path()
    _run_modal(["run", f"{impl}::batches", "--tag", tag, "--cmd", "submit-missing-force", "--batch-tag", batch_tag], tag)


@cli.command()
@click.argument("tag")
def collect(tag: str):
    """Collect results from completed jobs."""
    impl = _get_impl_path()
    _run_modal(["run", f"{impl}::batches", "--tag", tag, "--cmd", "collect"], tag)


@cli.command()
@click.argument("tag")
def status(tag: str):
    """Show status of jobs."""
    impl = _get_impl_path()
    _run_modal(["run", f"{impl}::batches", "--tag", tag, "--cmd", "status"], tag)


@cli.command()
@click.argument("tag")
def stop(tag: str):
    """Stop running jobs and clean up dicts."""
    app_name = f"yubo_{tag}"

    # First stop the app
    click.echo(f"Stopping app: {app_name}")
    stop_result = subprocess.run(["modal", "app", "stop", app_name])
    if stop_result.returncode != 0:
        click.echo(f"Warning: modal app stop returned {stop_result.returncode}")

    # Then clean up dicts
    impl = _get_impl_path()
    env = os.environ.copy()
    env["MODAL_TAG"] = tag
    click.echo(f"Cleaning up dicts for tag: {tag}")
    subprocess.run(["modal", "run", f"{impl}::batches", "--tag", tag, "--cmd", "stop"], env=env)


@cli.command("clean-up")
@click.argument("tag")
def clean_up(tag: str):
    """Clean up dicts without stopping jobs."""
    impl = _get_impl_path()
    _run_modal(["run", f"{impl}::batches", "--tag", tag, "--cmd", "clean_up"], tag)


if __name__ == "__main__":
    cli()
