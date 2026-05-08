import numpy as np


def _pareto_mask_max(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    assert y.ndim == 2, y.shape
    n = y.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        yi = y[i]
        ge = np.all(yi >= y, axis=1)
        gt = np.any(yi > y, axis=1)
        dom = ge & gt
        dom[i] = False
        keep[dom] = False
    return keep


def _pareto_mask_min(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    assert y.ndim == 2, y.shape
    n = y.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        yi = y[i]
        le = np.all(yi <= y, axis=1)
        lt = np.any(yi < y, axis=1)
        dom = le & lt
        dom[i] = False
        keep[dom] = False
    return keep
