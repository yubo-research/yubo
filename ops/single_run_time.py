#!/usr/bin/env python3
"""CLI for Modal BO timing sweep (`experiments/modal_timing_sweep.py`).

Examples:
  ./ops/single_run_time.py deploy
  ./ops/single_run_time.py submit mybatch --prep experiments.batch_preps.prep_tlunar
  ./ops/single_run_time.py progress --prep experiments.batch_preps.prep_timing_sweep
  ./ops/single_run_time.py collect
  ./ops/single_run_time.py stop
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODAL_APP_REL = Path("experiments") / "modal_timing_sweep.py"


def _add_repo_root_to_syspath() -> None:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


def _require_modal() -> None:
    try:
        import modal  # noqa: F401
    except ImportError as e:
        raise click.ClickException("modal is not installed; run: pip install modal") from e


def _load_prep(prep: str):
    if "." not in prep:
        raise click.ClickException(f"prep must be 'module.path.function_name' (got {prep!r}). Example: experiments.batch_preps.prep_tlunar")
    module_path, func_name = prep.rsplit(".", 1)
    _add_repo_root_to_syspath()
    try:
        module = __import__(module_path, fromlist=[func_name])
    except ImportError as e:
        raise click.ClickException(f"cannot import module {module_path!r}: {e}") from e
    fn = getattr(module, func_name, None)
    if fn is None:
        raise click.ClickException(f"module {module_path!r} has no attribute {func_name!r}")
    return fn


def run_modal_deploy() -> None:
    """Run ``modal deploy`` for ``experiments/modal_timing_sweep.py``."""
    script = _REPO_ROOT / _MODAL_APP_REL
    if not script.is_file():
        raise click.ClickException(f"Modal app not found: {script}")
    click.echo(f"Deploying {script.relative_to(_REPO_ROOT)} …")
    subprocess.run(
        ["modal", "deploy", str(script)],
        cwd=str(_REPO_ROOT),
        check=True,
    )


def run_modal_submit(batch_tag: str, prep: str, results_dir: str, force: bool) -> None:
    """Load prep(results_dir) and submit jobs to the timing-sweep Modal app."""
    from experiments.experiment_sampler import ExperimentConfig

    _require_modal()
    prep_fn = _load_prep(prep)
    configs = prep_fn(results_dir)
    if not isinstance(configs, list):
        raise click.ClickException(f"prep must return a list of ExperimentConfig, got {type(configs).__name__}")
    for i, cfg in enumerate(configs):
        if not isinstance(cfg, ExperimentConfig):
            raise click.ClickException(f"prep item {i} must be ExperimentConfig, got {type(cfg).__name__}")
        if int(cfg.num_reps) != 1:
            raise click.ClickException(f"timing sweep requires num_reps==1 per config (item {i} has num_reps={cfg.num_reps})")
    from experiments.modal_timing_sweep import submit_configs

    submit_configs(batch_tag, configs, force=force)


def run_prep_progress(prep: str, results_dir: str) -> None:
    """Print how many local traces from prep(results_dir) are done vs still missing."""
    from experiments.experiment_sampler import ExperimentConfig, count_local_trace_jobs

    prep_fn = _load_prep(prep)
    configs = prep_fn(results_dir)
    if not isinstance(configs, list):
        raise click.ClickException(f"prep must return a list of ExperimentConfig, got {type(configs).__name__}")
    for i, cfg in enumerate(configs):
        if not isinstance(cfg, ExperimentConfig):
            raise click.ClickException(f"prep item {i} must be ExperimentConfig, got {type(cfg).__name__}")
    n_done, n_left, n_total = count_local_trace_jobs(configs)
    click.echo(f"complete: {n_done}  remaining: {n_left}  total: {n_total}")


def run_modal_collect() -> None:
    """Collect timing-sweep results from Modal into local trace dirs."""
    _require_modal()
    from experiments.modal_timing_sweep import collect

    collect()


def run_modal_stop() -> None:
    """Stop the deployed timing-sweep app and delete its Modal Dicts (via ``modal run … clean_up``).

    Mirrors ``ops/stop_yubo.sh``: ``modal app stop <app>`` then ``modal run … --cmd=clean_up``.
    ``modal app stop`` uses ``check=False`` so a missing or idle app does not abort the cleanup step.
    """
    from experiments.modal_timing_sweep import _APP_NAME as timing_sweep_app_name

    script = _REPO_ROOT / _MODAL_APP_REL
    r_stop = subprocess.run(
        ["modal", "app", "stop", timing_sweep_app_name],
        cwd=str(_REPO_ROOT),
        check=False,
    )
    if r_stop.returncode != 0:
        click.echo(
            f"modal app stop exited {r_stop.returncode} (app may already be stopped); continuing to clean_up",
            err=True,
        )
    subprocess.run(
        ["modal", "run", str(script), "--cmd", "clean_up"],
        cwd=str(_REPO_ROOT),
        check=True,
    )


@click.group()
def cli() -> None:
    _add_repo_root_to_syspath()
    import ops.catalog  # noqa: F401 — integrate ops into import graph for static tools


@cli.command("deploy")
def deploy_cmd() -> None:
    """Deploy the timing-sweep Modal app (yubo_timing_sweep)."""
    run_modal_deploy()


@cli.command("submit")
@click.argument("batch_tag", type=str)
@click.option(
    "--prep",
    type=str,
    required=True,
    help="Prep function import path, e.g. experiments.batch_preps.prep_tlunar",
)
@click.option(
    "--results-dir",
    type=str,
    default="results",
    show_default=True,
    help="First argument passed to prep(results_dir)",
)
@click.option("--force", is_flag=True, help="Resubmit jobs already in the submitted dict")
def submit_cmd(batch_tag: str, prep: str, results_dir: str, force: bool) -> None:
    """Submit jobs from prep(results_dir) under BATCH_TAG."""
    run_modal_submit(batch_tag, prep, results_dir, force)


@cli.command("progress")
@click.option(
    "--prep",
    type=str,
    required=True,
    help="Prep function import path (same as submit), e.g. experiments.batch_preps.prep_timing_sweep",
)
@click.option(
    "--results-dir",
    type=str,
    default="results",
    show_default=True,
    help="First argument passed to prep(results_dir)",
)
def progress_cmd(prep: str, results_dir: str) -> None:
    """Count local traces: complete vs remaining (same paths as mk_replicates / submit)."""
    run_prep_progress(prep, results_dir)


@cli.command("collect")
def collect_cmd() -> None:
    """Pull finished results from Modal and write traces + *-summary.json locally."""
    run_modal_collect()


@cli.command("stop")
def stop_cmd() -> None:
    """``modal app stop`` the timing-sweep app, then ``modal run … clean_up`` (same idea as stop_yubo.sh)."""
    run_modal_stop()


if __name__ == "__main__":
    cli()
