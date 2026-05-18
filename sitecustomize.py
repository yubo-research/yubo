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


try:
    from gpytorch.lazy.lazy_evaluated_kernel_tensor import LazyEvaluatedKernelTensor

    if not hasattr(LazyEvaluatedKernelTensor, "add_diag") and hasattr(LazyEvaluatedKernelTensor, "add_diagonal"):
        LazyEvaluatedKernelTensor.add_diag = LazyEvaluatedKernelTensor.add_diagonal
except Exception:
    pass


try:
    from linear_operator.operators._linear_operator import LinearOperator

    if not hasattr(LinearOperator, "evaluate") and hasattr(LinearOperator, "to_dense"):
        LinearOperator.evaluate = LinearOperator.to_dense
    if not hasattr(LinearOperator, "inv_matmul") and hasattr(LinearOperator, "solve"):
        LinearOperator.inv_matmul = LinearOperator.solve
except Exception:
    pass
