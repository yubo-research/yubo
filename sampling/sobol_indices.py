import numpy as np


def calculate_sobol_indices_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.ndim == 2
    n, d = x.shape
    assert d > 0
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    assert y.ndim == 1
    assert y.shape[0] == n
    if n < 9:
        return np.ones(d, dtype=float)
    mu = np.mean(y)
    vy = np.var(y, ddof=0)
    if not np.isfinite(vy) or vy <= 0.0:
        return np.ones(d, dtype=float)
    B = 10 if n >= 30 else 3
    order = np.argsort(x, axis=0, kind="mergesort")
    ranks = np.empty_like(order)
    row_idx = np.arange(n)[:, None]
    col_idx = np.arange(d)[None, :]
    ranks[order, col_idx] = row_idx
    idx = (ranks * B) // n
    oh = np.eye(B, dtype=float)[idx]
    counts = oh.sum(axis=0)
    sums = (oh * y[:, None, None]).sum(axis=0)
    mu_b = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    p_b = counts / float(n)
    diff = mu_b - mu
    S = (p_b * (diff * diff)).sum(axis=1) / vy
    # S = np.clip(S, 0.0, 1.0)
    return S
