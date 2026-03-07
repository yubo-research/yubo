#!/usr/bin/env python3
import json
import shutil
import sys
from pathlib import Path

import click


def _add_repo_root_to_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _find_experiments(results_dir: Path):
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        config_path = exp_dir / "config.json"
        if not config_path.exists():
            continue
        yield exp_dir


def _load_config(exp_dir: Path) -> dict:
    config_path = exp_dir / "config.json"
    with open(config_path) as f:
        return json.load(f)


def _count_traces(exp_dir: Path) -> tuple[int, int]:
    traces_dir = exp_dir / "traces"
    if not traces_dir.exists():
        return 0, 0
    done = 0
    total = 0
    for f in traces_dir.iterdir():
        if f.suffix == ".jsonl":
            total += 1
            if f.with_suffix(".done").exists():
                done += 1
    return done, total


def _resolve_exp_dir(results_dir: Path, exp_hash: str) -> Path:
    exp_hash_path = Path(exp_hash)
    if exp_hash_path.is_absolute():
        raise click.ClickException(f"exp_hash must be a relative path, got: {exp_hash}")
    if ".." in exp_hash_path.parts:
        raise click.ClickException(f"exp_hash must not contain '..', got: {exp_hash}")

    results_dir_resolved = results_dir.resolve()
    exp_dir_resolved = (results_dir / exp_hash_path).resolve()

    if exp_dir_resolved == results_dir_resolved or results_dir_resolved not in exp_dir_resolved.parents:
        raise click.ClickException(f"exp_hash must point to a subdirectory of results_dir: results_dir={results_dir} exp_hash={exp_hash}")

    return exp_dir_resolved


def _resolve_exp_dirs(results_dir: Path, exp_hashes: tuple[str, ...]) -> list[Path]:
    if not exp_hashes:
        raise click.ClickException("Must provide at least one EXP_HASH")
    exp_dirs: list[Path] = []
    seen: set[Path] = set()
    for exp_hash in exp_hashes:
        exp_dir = _resolve_exp_dir(results_dir, exp_hash)
        if exp_dir in seen:
            continue
        seen.add(exp_dir)
        exp_dirs.append(exp_dir)
    return exp_dirs


@click.group()
def _cli():
    # Ensure this CLI is "integrated" into the repo's import graph for
    # static analysis tools (and allow future internal imports).
    _add_repo_root_to_syspath()
    import ops.catalog  # noqa: F401

    pass


cli = _cli


@_cli.command()
@click.argument("results_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--verbose", "-v", is_flag=True, help="Show full config")
def ls(results_dir: Path, verbose: bool):
    experiments = list(_find_experiments(results_dir))
    if not experiments:
        click.echo(f"No experiments found in {results_dir}")
        return

    for exp_dir in experiments:
        config = _load_config(exp_dir)
        done, total = _count_traces(exp_dir)

        status = "✓" if done == total and total > 0 else f"{done}/{total}"
        opt_name = config.get("opt_name", "?")
        env_tag = config.get("env_tag", "?")
        num_rounds = config.get("num_rounds", "?")
        num_arms = config.get("num_arms", "?")

        click.echo(f"{exp_dir.name}  [{status}]  {opt_name:20s}  {env_tag:20s}  arms={num_arms}  rounds={num_rounds}  reps={total}")

        if verbose:
            for k, v in sorted(config.items()):
                click.echo(f"    {k}: {v}")


@_cli.command()
@click.argument("exp_path", type=click.Path(exists=True, path_type=Path))
@click.option("--trace", "-t", type=int, default=0, help="Trace index to show")
@click.option("--head", "-n", type=int, default=None, help="Show first N lines")
@click.option("--config-only", "-c", is_flag=True, help="Show only config")
def cat(exp_path: Path, trace: int, head: int, config_only: bool):
    if exp_path.suffix == ".jsonl":
        trace_file = exp_path
        exp_dir = exp_path.parent.parent
    else:
        exp_dir = exp_path
        trace_file = exp_dir / "traces" / f"{trace:05d}.jsonl"

    config_path = exp_dir / "config.json"
    if config_path.exists():
        click.echo("=== config.json ===")
        with open(config_path) as f:
            config = json.load(f)
            click.echo(json.dumps(config, indent=2))

    if config_only:
        return

    if not trace_file.exists():
        click.echo(f"\nTrace file not found: {trace_file}", err=True)
        sys.exit(1)

    click.echo(f"\n=== {trace_file.name} ===")
    done_marker = trace_file.with_suffix(".done")
    if done_marker.exists():
        click.echo("Status: DONE")
    else:
        click.echo("Status: IN PROGRESS")

    click.echo("")
    with open(trace_file) as f:
        for i, line in enumerate(f):
            if head is not None and i >= head:
                click.echo(f"... ({head} of {sum(1 for _ in open(trace_file))} lines shown)")
                break
            record = json.loads(line)
            i_iter = record.get("i_iter", "?")
            rreturn = record.get("rreturn", "?")
            dt_prop = record.get("dt_prop", 0)
            dt_eval = record.get("dt_eval", 0)
            if isinstance(rreturn, float):
                click.echo(f"  {i_iter:4d}  ret={rreturn:10.4f}  dt_prop={dt_prop:.4f}  dt_eval={dt_eval:.4f}")
            else:
                click.echo(f"  {i_iter:4d}  ret={rreturn}  dt_prop={dt_prop}  dt_eval={dt_eval}")


@_cli.command()
@click.argument("results_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("exp_hashes", type=str, nargs=-1, required=True)
@click.option("--force", "-f", is_flag=True, help="Don't prompt for confirmation")
def rm(results_dir: Path, exp_hashes: tuple[str, ...], force: bool):
    exp_dirs = _resolve_exp_dirs(results_dir, exp_hashes)

    metas: list[tuple[Path, dict, int, int]] = []
    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            click.echo(f"Experiment directory not found: {exp_dir}", err=True)
            sys.exit(1)

        config_path = exp_dir / "config.json"
        if not config_path.exists():
            click.echo(f"Not an experiment directory: {exp_dir}", err=True)
            sys.exit(1)

        config = _load_config(exp_dir)
        done, total = _count_traces(exp_dir)
        metas.append((exp_dir, config, done, total))

    for exp_dir, config, done, total in metas:
        click.echo(f"Experiment: {exp_dir.name}")
        click.echo(f"  opt_name: {config.get('opt_name', '?')}")
        click.echo(f"  env_tag: {config.get('env_tag', '?')}")
        click.echo(f"  traces: {done}/{total} done")

    if not force:
        if not click.confirm(f"\nDelete {len(metas)} experiment(s)?"):
            click.echo("Aborted.")
            return

    for exp_dir, _, _, _ in metas:
        shutil.rmtree(exp_dir)
        click.echo(f"Deleted {exp_dir}")


if __name__ == "__main__":
    cli()
