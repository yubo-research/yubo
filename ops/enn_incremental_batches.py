#!/usr/bin/env python3
"""Deploy/submit/collect/stop wrapper for ENN add-timing and fit-timing Modal batches."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from ops.modal_cli_common import (
    collect_to_output_dir,
    run_modal,
    stop_app_and_delete_dicts,
)

_EXP_TYPE = click.Choice(["add_method", "fit_method"], case_sensitive=False)


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


@cli.command(
    params=[
        click.Argument(["function_name"]),
        click.Argument(["rep_index"], type=int),
        click.Argument(
            ["index_driver"],
            type=click.Choice(["flat", "hnsw"], case_sensitive=False),
        ),
        click.Option(("-d", "--d", "d_dims"), type=int, default=10, show_default=True),
        click.Option(
            ("--problem-seed", "problem_seed"),
            type=int,
            default=17,
            show_default=True,
        ),
        click.Option(
            ("--num-reps", "num_reps"),
            type=int,
            default=10,
            show_default=True,
        ),
        click.Option(
            ("--output-dir", "output_dir"),
            type=click.Path(),
            default="results/enn_incremental",
            show_default=True,
        ),
        click.Option(
            ("--checkpoints", "checkpoint_csv"),
            default="1,3,10",
            show_default=True,
            help="Comma-separated checkpoint Ns; use --checkpoints '' for all defaults.",
        ),
        click.Option(("--force/--no-force", "force"), default=False, show_default=True),
    ],
)
def local(
    function_name: str,
    rep_index: int,
    index_driver: str,
    d_dims: int,
    problem_seed: int,
    num_reps: int,
    output_dir: str,
    checkpoint_csv: str,
    force: bool,
):
    """Run one incremental ENN job locally and write the result JSON."""
    if d_dims < 1:
        raise click.BadParameter("D must be positive")
    if rep_index < 0:
        raise click.BadParameter("REP_INDEX must be >= 0")
    if num_reps < 1:
        raise click.BadParameter("num-reps must be >= 1")

    _ensure_repo_imports()
    from analysis.fitting_time.evaluate import synthetic_benchmark_data_seed
    from analysis.fitting_time.evaluate_metrics import normalize_benchmark_function_name
    from analysis.fitting_time.fitting_time_enn_incremental import (
        EnnIncrementalIndexDriver,
        benchmark_enn_incremental_add_timing,
    )
    from experiments.modal_enn_incremental_batches_impl import (
        result_json_dest,
        result_to_payload,
    )

    fn = normalize_benchmark_function_name(function_name)
    driver = EnnIncrementalIndexDriver(index_driver.lower())
    checkpoints = _parse_checkpoint_csv(checkpoint_csv or None)
    dest = result_json_dest(
        output_dir,
        d=d_dims,
        function_name=fn,
        problem_seed=problem_seed,
        rep_index=rep_index,
        num_reps=num_reps,
        index_driver=driver,
    )
    if dest.exists() and not force:
        click.echo(f"skip existing {dest.resolve()}")
        return

    data_seed = synthetic_benchmark_data_seed(
        function_name=fn,
        problem_seed=problem_seed,
        rep_index=rep_index,
    )
    click.echo(
        f"running incremental ENN D={d_dims} fn={fn} problem_seed={problem_seed} "
        f"rep_index={rep_index} data_seed={data_seed} index_driver={driver.value} "
        f"checkpoints={checkpoints or 'default'}",
        err=True,
    )
    result = benchmark_enn_incremental_add_timing(
        D=d_dims,
        function_name=fn,
        problem_seed=data_seed,
        index_driver=driver,
        checkpoints=checkpoints,
    )
    payload = result_to_payload(
        result,
        problem_seed=problem_seed,
        data_seed=data_seed,
        rep_index=rep_index,
        num_reps=num_reps,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    import json

    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    click.echo(f"wrote {dest.resolve()}")


@cli.command(
    "local-fit",
    params=[
        click.Argument(["function_name"]),
        click.Argument(["n"], type=int),
        click.Argument(["rep_index"], type=int),
        click.Argument(
            ["index_driver"],
            type=click.Choice(["flat", "hnsw"], case_sensitive=False),
        ),
        click.Option(("-d", "--d", "d_dims"), type=int, default=10, show_default=True),
        click.Option(
            ("--problem-seed", "problem_seed"),
            type=int,
            default=17,
            show_default=True,
        ),
        click.Option(
            ("--num-reps", "num_reps"),
            type=int,
            default=10,
            show_default=True,
        ),
        click.Option(
            ("--output-dir", "output_dir"),
            type=click.Path(),
            default="results/enn_incremental",
            show_default=True,
        ),
        click.Option(("--force/--no-force", "force"), default=False, show_default=True),
    ],
)
def local_fit(
    function_name: str,
    n: int,
    rep_index: int,
    index_driver: str,
    d_dims: int,
    problem_seed: int,
    num_reps: int,
    output_dir: str,
    force: bool,
):
    """Run one ENN fit-timing job locally and write the fit-only result JSON."""
    if d_dims < 1:
        raise click.BadParameter("D must be positive")
    if n < 1:
        raise click.BadParameter("N must be >= 1")
    if rep_index < 0:
        raise click.BadParameter("REP_INDEX must be >= 0")
    if num_reps < 1:
        raise click.BadParameter("num-reps must be >= 1")

    _ensure_repo_imports()
    from analysis.fitting_time.evaluate import synthetic_benchmark_data_seed
    from analysis.fitting_time.evaluate_metrics import normalize_benchmark_function_name
    from analysis.fitting_time.fitting_time_enn_fit import benchmark_enn_fit_timing
    from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver
    from experiments import modal_enn_fit_batches as fit_batches

    fn = normalize_benchmark_function_name(function_name)
    driver = EnnIncrementalIndexDriver(index_driver.lower())
    dest = fit_batches.fit_result_json_dest(
        output_dir,
        d=d_dims,
        function_name=fn,
        n=n,
        problem_seed=problem_seed,
        rep_index=rep_index,
        num_reps=num_reps,
        index_driver=driver,
        normalize_function_name=normalize_benchmark_function_name,
    )
    if (
        dest.exists()
        and not force
        and fit_batches.fit_result_json_complete(
            dest,
            n,
            d=d_dims,
            function_name=fn,
            problem_seed=problem_seed,
            rep_index=rep_index,
            num_reps=num_reps,
            index_driver=driver,
            normalize_function_name=normalize_benchmark_function_name,
        )
    ):
        click.echo(f"skip existing {dest.resolve()}")
        return

    data_seed = synthetic_benchmark_data_seed(
        function_name=fn,
        problem_seed=problem_seed,
        rep_index=rep_index,
    )
    result = benchmark_enn_fit_timing(
        D=d_dims,
        function_name=fn,
        data_seed=data_seed,
        problem_seed=problem_seed,
        n=n,
        index_driver=driver,
    )
    payload = fit_batches.fit_result_to_payload(
        result,
        problem_seed=problem_seed,
        data_seed=data_seed,
        rep_index=rep_index,
        num_reps=num_reps,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    import json

    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    click.echo(f"wrote {dest.resolve()}")


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


if __name__ == "__main__":
    cli()
