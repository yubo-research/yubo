#!/usr/bin/env python

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    # When running as `./experiment/experiment.py`, sys.path[0] is this directory,
    # so we need to add the repository root to import `experiments.*`.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def _main() -> None:
    _ensure_repo_root_on_path()

    from experiments.experiment import cli

    cli()


main = _main


if __name__ == "__main__":
    main()
