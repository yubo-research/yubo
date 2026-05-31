from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def copy_vector(objective: Any, x: Any) -> Any:
    method = getattr(objective, "copy_vector", None)
    if callable(method):
        return method(x)
    copy = getattr(x, "copy", None)
    return copy() if callable(copy) else np.array(x, copy=True)


def stack_vectors(objective: Any, xs: Iterable[Any]) -> Any:
    method = getattr(objective, "stack_vectors", None)
    if callable(method):
        return method(tuple(xs))
    return np.stack(tuple(xs))


def zeros_vector(objective: Any, dim: int) -> Any:
    method = getattr(objective, "zeros_vector", None)
    if callable(method):
        return method(int(dim))
    return np.zeros(int(dim), dtype=np.float64)
