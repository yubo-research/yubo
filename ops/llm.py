#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import click


if sys.version_info < (3, 11):
    sys.stderr.write(
        "ops/llm.py requires Python >= 3.11. "
        "Run it from the yubo-hyperscalees env, for example: "
        "micromamba run -n yubo-hyperscalees ./ops/llm.py local <config>\n"
    )
    raise SystemExit(1)


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def _forward_to_llm_cli(extra_args: list[str]) -> None:
    _ensure_repo_root_on_path()
    import experiments.llm as llm_mod

    target = llm_mod.cli
    if extra_args:
        target.main(args=extra_args, prog_name="ops/llm.py", standalone_mode=False)
    else:
        target()


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def _cli(ctx: click.Context) -> None:
    _forward_to_llm_cli(list(ctx.args))


cli = _cli


def _main() -> None:
    cli(standalone_mode=True)


main = _main


if __name__ == "__main__":
    main()
