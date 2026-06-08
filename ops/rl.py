#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import click


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def _forward_to_rl_runner(extra_args: list[str]) -> None:
    _ensure_repo_root_on_path()
    from rl.runner import main as runner_main

    args = list(extra_args)
    if args and args[0] == "local":
        args = args[1:]
    if args and args[0] != "--config":
        args = ["--config", *args]
    runner_main(args)


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def _cli(ctx: click.Context) -> None:
    _forward_to_rl_runner(list(ctx.args))


cli = _cli


def main() -> None:
    cli(standalone_mode=True)


if __name__ == "__main__":
    main()
