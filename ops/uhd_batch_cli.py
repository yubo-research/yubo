import sys

import click

from ops.uhd_batch_core import _APP_NAME, _DEFAULT_RESULTS, _load_toml, _resolve_num_reps
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

_ORIG_BATCH_LOCAL = _batch_local
_ORIG_BATCH_MODAL = _batch_modal


def _batch_module():
    return sys.modules.get("ops.uhd_batch")


def _resolve_batch_local():
    local_fn = _batch_local
    if local_fn is not _ORIG_BATCH_LOCAL:
        return local_fn
    batch_mod = _batch_module()
    if batch_mod is None:
        return local_fn
    return getattr(batch_mod, "_batch_local", local_fn)


def _resolve_batch_modal():
    local_fn = _batch_modal
    if local_fn is not _ORIG_BATCH_MODAL:
        return local_fn
    batch_mod = _batch_module()
    if batch_mod is None:
        return local_fn
    return getattr(batch_mod, "_batch_modal", local_fn)


def _load_prep_configs(prep: str, results_dir: str) -> list[tuple[dict, int]]:
    module_path, func_name = prep.rsplit(".", 1)
    module = __import__(module_path, fromlist=[func_name])
    prep_fn = getattr(module, func_name)
    configs = prep_fn(results_dir)
    if not isinstance(configs, list):
        raise click.ClickException(f"{prep} must return list[tuple[dict, int]]")
    return configs


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
@click.option(
    "--num-reps",
    type=int,
    default=None,
    help="Number of replications; overrides [uhd].num_reps",
)
@click.option("--workers", type=int, default=1, help="Parallel workers")
@click.option("--results-dir", type=str, default=_DEFAULT_RESULTS, help="Results directory")
def local_cmd(config_toml: str, num_reps: int | None, workers: int, results_dir: str) -> None:
    """Run replications locally (subprocess pool, not Modal)."""
    cfg = _load_toml(config_toml)
    _resolve_batch_local()(cfg, _resolve_num_reps(cfg, num_reps), results_dir, workers)


@_cli.command(name="modal")
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option(
    "--num-reps",
    type=int,
    default=None,
    help="Number of replications; overrides [uhd].num_reps",
)
@click.option("--results-dir", type=str, default=_DEFAULT_RESULTS, help="Results directory")
def modal_cmd(config_toml: str, num_reps: int | None, results_dir: str) -> None:
    cfg = _load_toml(config_toml)
    _resolve_batch_modal()(cfg, _resolve_num_reps(cfg, num_reps), results_dir)


@_cli.command(name="cleanup")
def cleanup_cmd() -> None:
    import modal

    _require_modal()
    for name in ["uhd_batch_results", "uhd_batch_submitted"]:
        try:
            modal.Dict.delete(name)
            click.echo(f"Deleted dict: {name}")
        except Exception as e:
            click.echo(f"Delete failed for {name}: {e!r}")


@_cli.command(name="batch")
@click.argument("import_path", type=str)
@click.option("--results-dir", type=str, default=_DEFAULT_RESULTS, help="Results directory")
def batch_cmd(import_path: str, results_dir: str) -> None:
    _require_modal()
    _ensure_uhd_batch_app()
    configs = _load_prep_configs(import_path, results_dir)
    for cfg, n_reps in configs:
        _batch_modal(cfg, n_reps, results_dir)
