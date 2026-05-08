import click

from ops.uhd_batch_core import _DEFAULT_RESULTS, _load_toml
from ops.uhd_batch_local import _batch_local
from ops.uhd_batch_modal import (
    _batch_modal,
    _collect,
    _require_modal,
    _results_dict,
    _submitted_dict,
)


@click.group()
def _cli():
    pass


cli = _cli


@_cli.command(name="local")
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--num-reps", type=int, required=True, help="Number of replications")
@click.option("--workers", type=int, default=1, help="Parallel workers")
@click.option("--results-dir", type=str, default=_DEFAULT_RESULTS, help="Results directory")
def local_cmd(config_toml: str, num_reps: int, workers: int, results_dir: str) -> None:
    cfg = _load_toml(config_toml)
    _batch_local(cfg, num_reps, results_dir, workers)


@_cli.command(name="modal")
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--num-reps", type=int, required=True, help="Number of replications")
@click.option("--results-dir", type=str, default=_DEFAULT_RESULTS, help="Results directory")
def modal_cmd(config_toml: str, num_reps: int, results_dir: str) -> None:
    cfg = _load_toml(config_toml)
    _batch_modal(cfg, num_reps, results_dir)


@_cli.command(name="collect")
@click.option("--results-dir", type=str, default=_DEFAULT_RESULTS, help="Results directory")
def collect_cmd(results_dir: str) -> None:
    _collect(results_dir)


@_cli.command(name="status")
def status_cmd() -> None:
    _require_modal()
    rd = _results_dict()
    sd = _submitted_dict()
    click.echo(f"results_available = {rd.len()}")
    click.echo(f"submitted = {sd.len()}")


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

    module_path, func_name = import_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[func_name])
    prep_fn = getattr(module, func_name)

    configs = prep_fn(results_dir)
    total_submitted = 0

    for cfg, num_reps in configs:
        _batch_modal(cfg, num_reps, results_dir)
        total_submitted += num_reps

    click.echo(f"Batch complete: {len(configs)} configs, {total_submitted} total reps submitted")
