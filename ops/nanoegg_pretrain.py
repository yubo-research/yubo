#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import click


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def _forward_to_nanoegg_cli(extra_args: list[str]) -> None:
    _ensure_repo_root_on_path()
    import experiments.nanoegg_pretrain as nanoegg_pretrain_mod

    target = nanoegg_pretrain_mod.cli
    if extra_args:
        target.main(args=extra_args, prog_name="ops/nanoegg_pretrain.py", standalone_mode=False)
    else:
        target()


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def _cli(ctx: click.Context) -> None:
    _forward_to_nanoegg_cli(list(ctx.args))


cli = _cli


def _main() -> None:
    cli(standalone_mode=True)


main = _main


if __name__ == "__main__":
    main()
