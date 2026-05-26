#!/usr/bin/env python3
import sys
from pathlib import Path

import click
import tomllib


def _add_repo_root_to_syspath():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


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
def _cli():
    """Print catalogs of registry-like things."""


cli = _cli


@_cli.command()
def designers():
    """List all designer base names and their options."""
    _add_repo_root_to_syspath()
    from optimizer.designers import Designers

    d = Designers(_CatalogPolicy(), num_arms=1)
    for entry in d.catalog():
        click.echo(_describe(entry))


@_cli.command()
def policies():
    """List all available policy tags."""
    _add_repo_root_to_syspath()
    from policies.registry import list_policy_tags

    for tag in list_policy_tags():
        click.echo(tag)


@_cli.command()
def environments():
    _add_repo_root_to_syspath()
    from problems.benchmark_functions import all_benchmarks
    from problems.environment_spec import (
        _atari_env_specs,
        _dm_control_env_specs,
        _gym_env_specs,
    )

    fn_names = sorted(all_benchmarks().keys())
    other_envs = {"mopta08", "push", "leukemia", "dna", "rcv1"}
    env_names = sorted(set(_gym_env_specs.keys()) | set(_dm_control_env_specs.keys()) | set(_atari_env_specs.keys()) | other_envs)

    for name in fn_names:
        click.echo(f"f:{name}")
    for name in env_names:
        click.echo(name)


@_cli.command(name="jax-envs")
def jax_envs():
    """List first-class JAX env-family tags known to the runtime."""
    _add_repo_root_to_syspath()
    from problems.jax_env_core import supported_jax_env_tags

    for tag in supported_jax_env_tags():
        click.echo(tag)


@_cli.command(name="llm-envs")
def llm_envs():
    """List first-class LLM env tags."""
    _add_repo_root_to_syspath()
    from llm.registry import supported_llm_env_tags

    for tag in supported_llm_env_tags():
        click.echo(tag)


@_cli.command(name="pretrain-envs")
def pretrain_envs():
    """List first-class pretraining env tags."""
    _add_repo_root_to_syspath()
    from problems.pre_obj import (
        supported_hyperscalees_pretrain_env_tags,
        supported_nanoegg_pretrain_examples,
    )

    for tag in supported_hyperscalees_pretrain_env_tags():
        click.echo(tag)
    for env_tag, policy_tag in supported_nanoegg_pretrain_examples():
        click.echo(f"{env_tag} policy={policy_tag}")


@_cli.command()
def uhd():
    """List all UHD optimizers (for use with optimizer= in [uhd] config)."""
    for name in ("simple", "simple_be", "mezo", "mezo_be", "bszo"):
        click.echo(name)


@_cli.command(name="rl-algos")
def rl_algos():
    """List supported RL algorithms."""
    for name in ("ppo", "sac"):
        click.echo(name)


@_cli.command(name="rl-configs")
def rl_configs():
    """List supported RL config files."""
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "configs" / "rl").rglob("*.toml")):
        with path.open("rb") as f:
            data = tomllib.load(f)
        if "rl" in data:
            click.echo(path.relative_to(root).as_posix())


if __name__ == "__main__":
    cli()
