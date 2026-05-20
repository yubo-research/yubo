#!/usr/bin/env python3
"""Deploy/submit/collect/stop wrapper for ENN add-timing and fit-timing Modal batches."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from ops.enn_incremental_batches_local import register_local_commands
from ops.modal_cli_common import (
    collect_to_output_dir,
    run_modal,
    stop_app_and_delete_dicts,
)

_EXP_TYPE = click.Choice(["add_method", "fit_method", "fit_ind"], case_sensitive=False)


def _exp_type_argument() -> click.Argument:
    return click.Argument(["exp_type"], type=_EXP_TYPE, metavar="EXP_TYPE")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_repo_imports() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _get_impl_path() -> str:
    return "experiments/modal_enn_incremental_batches_impl.py"


def _modal_tag(exp_type: str, tag: str) -> str:
    return f"{exp_type.lower()}-{tag}"


def _get_app_name(tag: str) -> str:
    return f"yubo-enn-incremental-{tag}"


def _modal_dict_names(tag: str) -> tuple[str, str]:
    return (
        f"enn_incremental_results_{tag}",
        f"enn_incremental_submitted_{tag}",
    )


def _run_modal(args: list[str], tag: str) -> None:
    run_modal(args, tag, run=subprocess.run)


def _resolve_checkpoints(raw: str | None) -> tuple[int, ...]:
    from experiments.enn_batch_job_params import enn_batch_checkpoint_ns

    if raw is not None and not str(raw).strip():
        raw = None
    parsed = _parse_checkpoint_csv(raw)
    if parsed is None:
        return enn_batch_checkpoint_ns()
    return parsed


def _parse_checkpoint_csv(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise click.BadParameter("checkpoints must be a comma-separated list of ints")
    try:
        checkpoints = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise click.BadParameter("checkpoints must be a comma-separated list of ints") from exc
    prev = 0
    for checkpoint in checkpoints:
        if checkpoint <= prev:
            raise click.BadParameter("checkpoints must be strictly increasing")
        prev = checkpoint
    return checkpoints


@click.group()
def cli():
    pass


@cli.command()
@click.argument("exp_type", type=_EXP_TYPE, metavar="EXP_TYPE")
@click.argument("tag")
def deploy(exp_type: str, tag: str):
    """Deploy the incremental ENN Modal app for this experiment type and tag."""
    _run_modal(["deploy", _get_impl_path()], _modal_tag(exp_type, tag))


@cli.command(
    params=[
        _exp_type_argument(),
        click.Argument(["tag"]),
        click.Option(["--output-dir"], default="results/enn_incremental", show_default=True),
        click.Option(
            ["--index-driver"],
            type=click.Choice(["flat", "hnsw", "all"], case_sensitive=False),
            default="all",
            show_default=True,
        ),
        click.Option(["--num-reps"], default=10, type=int, show_default=True),
        click.Option(("-d", "--d", "d_dims"), type=int, default=10, show_default=True),
        click.Option(
            ("--problem-seed", "problem_seed"),
            type=int,
            default=17,
            show_default=True,
        ),
    ],
)
def submit(
    exp_type: str,
    tag: str,
    output_dir: str,
    index_driver: str,
    num_reps: int,
    d_dims: int,
    problem_seed: int,
):
    """Submit missing ENN jobs for the fixed benchmark targets."""
    if d_dims < 1:
        raise click.BadParameter("D must be positive")
    if num_reps < 1:
        raise click.BadParameter("num-reps must be >= 1")
    impl = _get_impl_path()
    modal_tag = _modal_tag(exp_type, tag)
    _run_modal(
        [
            "run",
            f"{impl}::batches",
            "--tag",
            modal_tag,
            "--cmd",
            "submit",
            "--output-dir",
            output_dir,
            "--index-driver",
            index_driver.lower(),
            "--num-reps",
            str(num_reps),
            "--d",
            str(d_dims),
            "--problem-seed",
            str(problem_seed),
        ],
        modal_tag,
    )


@cli.command(
    "submit-force",
    params=[
        _exp_type_argument(),
        click.Argument(["tag"]),
        click.Option(["--output-dir"], default="results/enn_incremental", show_default=True),
        click.Option(
            ["--index-driver"],
            type=click.Choice(["flat", "hnsw", "all"], case_sensitive=False),
            default="all",
            show_default=True,
        ),
        click.Option(["--num-reps"], default=10, type=int, show_default=True),
        click.Option(("-d", "--d", "d_dims"), type=int, default=10, show_default=True),
        click.Option(
            ("--problem-seed", "problem_seed"),
            type=int,
            default=17,
            show_default=True,
        ),
    ],
)
def submit_force(
    exp_type: str,
    tag: str,
    output_dir: str,
    index_driver: str,
    num_reps: int,
    d_dims: int,
    problem_seed: int,
):
    """Force resubmit all pending ENN jobs, including those already marked submitted."""
    if d_dims < 1:
        raise click.BadParameter("D must be positive")
    if num_reps < 1:
        raise click.BadParameter("num-reps must be >= 1")
    impl = _get_impl_path()
    modal_tag = _modal_tag(exp_type, tag)
    _run_modal(
        [
            "run",
            f"{impl}::batches",
            "--tag",
            modal_tag,
            "--cmd",
            "submit-force",
            "--output-dir",
            output_dir,
            "--index-driver",
            index_driver.lower(),
            "--num-reps",
            str(num_reps),
            "--d",
            str(d_dims),
            "--problem-seed",
            str(problem_seed),
        ],
        modal_tag,
    )


@cli.command()
@click.argument("exp_type", type=_EXP_TYPE, metavar="EXP_TYPE")
@click.argument("tag")
@click.option(
    "--output-dir",
    default="results/enn_incremental",
    show_default=True,
)
def collect(exp_type: str, tag: str, output_dir: str):
    """Collect completed results to the local output directory."""
    collect_to_output_dir(_get_impl_path(), _modal_tag(exp_type, tag), output_dir, run=subprocess.run)


@cli.command()
@click.argument("exp_type", type=_EXP_TYPE, metavar="EXP_TYPE")
@click.argument("tag")
def status(exp_type: str, tag: str):
    """Show submitted/result dict sizes for this tag."""
    impl = _get_impl_path()
    modal_tag = _modal_tag(exp_type, tag)
    _run_modal(["run", f"{impl}::batches", "--tag", modal_tag, "--cmd", "status"], modal_tag)


@cli.command()
@click.argument("exp_type", type=_EXP_TYPE, metavar="EXP_TYPE")
@click.argument("tag")
def stop(exp_type: str, tag: str):
    """Stop the app and clean up the backing Modal dicts."""
    modal_tag = _modal_tag(exp_type, tag)
    stop_app_and_delete_dicts(
        app_name=_get_app_name(modal_tag),
        dict_names=_modal_dict_names(modal_tag),
        run=subprocess.run,
    )


register_local_commands(
    cli,
    resolve_checkpoints=_resolve_checkpoints,
    ensure_repo_imports=_ensure_repo_imports,
)


if __name__ == "__main__":
    cli()
