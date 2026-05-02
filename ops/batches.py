#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import click


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


@click.command()
@click.argument("batch_tag", type=str)
@click.option("--max-parallel", default=5, show_default=True, type=int, help="Max concurrent configs per batch.")
@click.option("--dry-run", is_flag=True, help="Print scheduled runs without executing.")
@click.option("--results-dir", default="results", show_default=True, type=str, help="Results root passed to prep_* functions.")
def _cli(batch_tag: str, max_parallel: int, dry_run: bool, results_dir: str) -> None:
    _ensure_repo_root_on_path()
    from experiments.batches_impl import run_from_batch_tag

    run_from_batch_tag(
        batch_tag,
        max_parallel=max_parallel,
        dry_run=dry_run,
        results_dir=results_dir,
    )


cli = _cli


def _main() -> None:
    cli()


main = _main


if __name__ == "__main__":
    main()
