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
    _run_modal(["run", f"{impl}::batches", tag, "submit-missing", batch_tag], tag)


@cli.command("submit-force")
@click.argument("tag")
@click.argument("batch_tag")
def submit_force(tag: str, batch_tag: str):
    """Force resubmit all jobs for the batch."""
    impl = _get_impl_path()
    _run_modal(["run", f"{impl}::batches", tag, "submit-missing-force", batch_tag], tag)


@cli.command()
@click.argument("tag")
def collect(tag: str):
    """Collect results from completed jobs."""
    impl = _get_impl_path()
    _run_modal(["run", f"{impl}::batches", tag, "collect"], tag)


@cli.command()
@click.argument("tag")
def status(tag: str):
    """Show status of jobs."""
    impl = _get_impl_path()
    _run_modal(["run", f"{impl}::batches", tag, "status"], tag)


@cli.command()
@click.argument("tag")
def stop(tag: str):
    """Stop running jobs and clean up dicts."""
    impl = _get_impl_path()
    _run_modal(["run", f"{impl}::batches", tag, "stop"], tag)


@cli.command("clean-up")
@click.argument("tag")
def clean_up(tag: str):
    """Clean up dicts without stopping jobs."""
    impl = _get_impl_path()
    _run_modal(["run", f"{impl}::batches", tag, "clean_up"], tag)


if __name__ == "__main__":
    cli()
