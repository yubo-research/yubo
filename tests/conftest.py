"""Pytest configuration: ensure tests/ is on sys.path for co-located helper modules."""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_p = str(_TESTS_DIR)
if _p not in sys.path:
    # Append (do not prepend): prepending tests/ can shadow repo packages and break
    # imports that rely on repository-root-first resolution (e.g. ops.experiment forwarding).
    sys.path.append(_p)
