#!/usr/bin/env python3
import sys
from pathlib import Path

import click

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _add_repo_root_to_syspath():
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


class _CatalogPolicy:
    problem_seed = 0

    def num_params(self):
        return 1


def _format_option(opt) -> str:
    allowed = ""
    if opt.allowed_values:
        allowed = " in {" + ",".join(opt.allowed_values) + "}"
    return f"{opt.name}:{opt.value_type}{allowed}"


def _describe(entry) -> str:
    if not entry.options:
        return f"{entry.base_name}"
    opts = ", ".join(_format_option(o) for o in entry.options)
    return f"{entry.base_name} - {opts}"


@click.group()
def cli():
    """Print catalogs of registry-like things."""


@cli.command()
def designers():
    """List all designer base names and their options."""
    _add_repo_root_to_syspath()
    from optimizer.designers import Designers

    d = Designers(_CatalogPolicy(), num_arms=1)
    for entry in d.catalog():
        click.echo(_describe(entry))


@cli.command()
def environments():
    _add_repo_root_to_syspath()
    from problems.benchmark_functions import all_benchmarks
    from problems.env_conf import _gym_env_confs

    fn_names = sorted(all_benchmarks().keys())
    other_envs = {"mopta08", "push", "leukemia", "dna", "rcv1"}
    env_names = sorted(set(_gym_env_confs.keys()) | other_envs)

    for name in fn_names:
        click.echo(f"f:{name}")
    for name in env_names:
        click.echo(name)


@cli.command(name="data-commands")
def data_commands():
    """List available `ops/data.py` subcommands."""
    _add_repo_root_to_syspath()
    import ops.data as _ops_data

    for cmd_name in sorted(_ops_data.cli.commands.keys()):
        click.echo(cmd_name)


if __name__ == "__main__":
    cli()
