#!/usr/bin/env python

from __future__ import annotations

import sys
from pathlib import Path

import click


def _ensure_repo_root_on_path() -> None:
    # When running as `./experiment/experiment.py`, sys.path[0] is this directory,
    # so we need to add the repository root to import `experiments.*`.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def _forward_to_experiments_cli(extra_args: list[str]) -> None:
    _ensure_repo_root_on_path()
    import experiments.experiment as experiment_mod

    target = experiment_mod.cli
    if extra_args:
        target.main(args=extra_args, prog_name="ops/experiment.py", standalone_mode=False)
    else:
        target()


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def _cli(ctx: click.Context) -> None:
    _forward_to_experiments_cli(list(ctx.args))


cli = _cli


def _main() -> None:
    cli(standalone_mode=True)


main = _main


if __name__ == "__main__":
    main()
