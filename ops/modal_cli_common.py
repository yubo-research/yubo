"""Shared helpers for small Modal CLI wrappers."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Sequence

import click

RunFn = Callable[..., subprocess.CompletedProcess]


def run_modal(args: Sequence[str], tag: str, *, run: RunFn = subprocess.run) -> None:
    env = os.environ.copy()
    env["MODAL_TAG"] = tag
    cmd = ["modal", *args]
    click.echo(f"Running: {' '.join(cmd)} (MODAL_TAG={tag})")
    result = run(cmd, env=env)
    sys.exit(result.returncode)


def run_modal_entrypoint(
    impl: str,
    tag: str,
    cmd: str,
    *,
    extra_args: Sequence[str] = (),
    run: RunFn = subprocess.run,
) -> None:
    run_modal(
        ["run", f"{impl}::batches", "--tag", tag, "--cmd", cmd, *extra_args],
        tag,
        run=run,
    )


def collect_to_output_dir(
    impl: str,
    tag: str,
    output_dir: str,
    *,
    run: RunFn = subprocess.run,
) -> None:
    run_modal_entrypoint(
        impl,
        tag,
        "collect",
        extra_args=["--output-dir", output_dir],
        run=run,
    )


def stop_app_and_delete_dicts(
    *,
    app_name: str,
    dict_names: Sequence[str],
    run: RunFn = subprocess.run,
) -> None:
    exit_code = 0
    click.echo(f"Stopping app: {app_name}")
    stop_result = run(["modal", "app", "stop", app_name])
    if stop_result.returncode != 0:
        click.echo(f"Warning: modal app stop returned {stop_result.returncode}")
        exit_code = 1

    for name in dict_names:
        click.echo(f"Deleting Modal dict: {name}")
        del_result = run(["modal", "dict", "delete", "--yes", "--allow-missing", name])
        if del_result.returncode != 0:
            click.echo(f"Warning: modal dict delete returned {del_result.returncode} for {name!r}")
            exit_code = 1

    if exit_code != 0:
        sys.exit(exit_code)
