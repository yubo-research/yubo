"""Repository-wide compatibility shims for pinned third-party stacks.

This file is imported automatically by Python when the repo root is on
``sys.path``. Keep the patches minimal and version-tolerant.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    if not hasattr(np, "NaN"):
        np.NaN = np.nan

    if not hasattr(np, "Inf"):
        np.Inf = np.inf


def _patch_gpytorch_lazy(module) -> None:
    lazy_tensor = getattr(module, "LazyEvaluatedKernelTensor", None)
    if lazy_tensor is not None and not hasattr(lazy_tensor, "add_diag") and hasattr(lazy_tensor, "add_diagonal"):
        lazy_tensor.add_diag = lazy_tensor.add_diagonal


def _patch_linear_operator(module) -> None:
    linear_operator = getattr(module, "LinearOperator", None)
    if linear_operator is None:
        return
    if not hasattr(linear_operator, "evaluate") and hasattr(linear_operator, "to_dense"):
        linear_operator.evaluate = linear_operator.to_dense
    if not hasattr(linear_operator, "inv_matmul") and hasattr(linear_operator, "solve"):
        linear_operator.inv_matmul = linear_operator.solve


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, loader, patch):
        self._loader = loader
        self._patch = patch

    def create_module(self, spec):
        create_module = getattr(self._loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module) -> None:
        self._loader.exec_module(module)
        try:
            self._patch(module)
        except Exception:
            pass


class _PatchFinder(importlib.abc.MetaPathFinder):
    def __init__(self, fullname: str, patch) -> None:
        self.fullname = fullname
        self.patch = patch

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.fullname:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _PatchLoader(spec.loader, self.patch)
        return spec


def _patch_when_imported(fullname: str, patch) -> None:
    module = sys.modules.get(fullname)
    if module is not None:
        try:
            patch(module)
        except Exception:
            pass
        return
    sys.meta_path.insert(0, _PatchFinder(fullname, patch))


_patch_when_imported("gpytorch.lazy.lazy_evaluated_kernel_tensor", _patch_gpytorch_lazy)
_patch_when_imported("linear_operator.operators._linear_operator", _patch_linear_operator)
