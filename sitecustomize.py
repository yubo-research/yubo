"""Repository-wide compatibility shims for pinned third-party stacks.

This file is imported automatically by Python when the repo root is on
``sys.path``. Keep the patches minimal and version-tolerant.
"""

from __future__ import annotations

import os
import sys

_jax_platforms = os.environ.get("JAX_PLATFORMS", "")
if sys.platform == "darwin" and os.environ.get("CODEX_SANDBOX") and _jax_platforms.split(",")[0] == "mps":
    os.environ["JAX_PLATFORMS"] = "cpu"

_mujoco_gl = os.environ.get("MUJOCO_GL", "").lower().strip()
if sys.platform == "darwin" and _mujoco_gl in {"egl", "glx", "osmesa"}:
    os.environ["MUJOCO_GL"] = "cgl"

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    if not hasattr(np, "NaN"):
        np.NaN = np.nan

    if not hasattr(np, "Inf"):
        np.Inf = np.inf

    try:
        np.asarray(0, copy=None)
    except TypeError:
        _np_asarray = np.asarray

        def _asarray_copy_compat(a, dtype=None, order=None, *, copy=None, like=None):
            kwargs = {}
            if dtype is not None:
                kwargs["dtype"] = dtype
            if order is not None:
                kwargs["order"] = order
            if like is not None:
                kwargs["like"] = like
            arr = _np_asarray(a, **kwargs)
            if copy is True:
                return arr.copy()
            return arr

        np.asarray = _asarray_copy_compat

    try:
        import sklearn.tree._tree as _sklearn_tree
    except ImportError:
        _sklearn_tree = None

    if _sklearn_tree is not None and not hasattr(_sklearn_tree, "DTYPE"):
        _sklearn_tree.DTYPE = np.float32
