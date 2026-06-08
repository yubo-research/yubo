"""Repository-wide compatibility shims for pinned third-party stacks.

This file is imported automatically by Python when the repo root is on
``sys.path``. Keep the patches minimal and version-tolerant.
"""

from __future__ import annotations

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    if not hasattr(np, "NaN"):
        np.NaN = np.nan

    if not hasattr(np, "Inf"):
        np.Inf = np.inf
