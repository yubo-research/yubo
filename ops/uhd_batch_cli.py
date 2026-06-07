import click

from ops.uhd_batch_core import _APP_NAME, _DEFAULT_RESULTS, _load_toml
from ops.uhd_batch_local import _batch_local
from ops.uhd_batch_modal import (
    _batch_modal,
    _collect,
    _deploy_uhd_batch_app,
    _ensure_uhd_batch_app,
    _require_modal,
    _results_dict,
    _stop_uhd_batch,
    _submitted_dict,
)


@click.group()
def _cli():
    pass


cli = _cli


@_cli.command(name="deploy")
def deploy_cmd() -> None:
    """Deploy the UHD batch Modal app (yubo_uhd_batch)."""
    _deploy_uhd_batch_app()


@_cli.command(name="submit")
@click.option(
    "--prep",
    type=str,
    default=None,
    help="Prep import path, e.g. experiments.uhd_batch_preps.prep_uhd_batch_cheetah",
)
@click.option(
    "--config",
    "config_toml",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="Single [uhd] config TOML (use with --num-reps)",
)
@click.option("--num-reps", type=int, default=None, help="Replications for --config")
@click.option("--results-dir", type=str, default=_DEFAULT_RESULTS, show_default=True)
def submit_cmd(
    prep: str | None,
    config_toml: str | None,
    num_reps: int | None,
    results_dir: str,
) -> None:
    """Submit missing UHD batch jobs from a prep function or one config TOML."""
    _require_modal()
    _ensure_uhd_batch_app()

    if prep is not None and config_toml is not None:
        raise click.ClickException("Specify exactly one of --prep or --config")
    if prep is None and config_toml is None:
        raise click.ClickException("Specify --prep or --config")

    if prep is not None:
        configs = _load_prep_configs(prep, results_dir)
    else:
        if num_reps is None:
            raise click.ClickException("--num-reps is required with --config")
        if num_reps < 1:
            raise click.BadParameter("num-reps must be >= 1")
        configs = [(_load_toml(config_toml), num_reps)]

    total_submitted = 0
    for cfg, n_reps in configs:
        _batch_modal(cfg, n_reps, results_dir, ensure_deployed=False)
        total_submitted += n_reps

    click.echo(f"Submit complete: {len(configs)} configs, {total_submitted} total reps submitted")


@_cli.command(name="collect")
@click.option("--results-dir", type=str, default=_DEFAULT_RESULTS, show_default=True)
def collect_cmd(results_dir: str) -> None:
    """Pull finished results from Modal and write local trace JSONL files."""
    _collect(results_dir)


@_cli.command(name="status")
def status_cmd() -> None:
    """Show Modal result/submitted dict sizes."""
    _require_modal()
    rd = _results_dict()
    sd = _submitted_dict()
    click.echo(f"app = {_APP_NAME}")
    click.echo(f"results_available = {rd.len()}")
    click.echo(f"submitted = {sd.len()}")


@_cli.command(name="stop")
def stop_cmd() -> None:
    """Stop the Modal app and delete uhd_batch_* dicts."""
    _stop_uhd_batch()


@_cli.command(name="local")
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--num-reps", type=int, required=True, help="Number of replications")
@click.option("--workers", type=int, default=1, help="Parallel workers")
@click.option("--results-dir", type=str, default=_DEFAULT_RESULTS, help="Results directory")
def local_cmd(config_toml: str, num_reps: int, workers: int, results_dir: str) -> None:
    """Run replications locally (subprocess pool, not Modal)."""
    cfg = _load_toml(config_toml)
    _batch_local(cfg, num_reps, results_dir, workers)


def _load_prep_configs(prep: str, results_dir: str) -> list[tuple[dict, int]]:
    module_path, func_name = prep.rsplit(".", 1)
    module = __import__(module_path, fromlist=[func_name])
    prep_fn = getattr(module, func_name)
    configs = prep_fn(results_dir)
    if not isinstance(configs, list):
        raise click.ClickException(f"{prep} must return list[tuple[dict, int]]")
    return configs
