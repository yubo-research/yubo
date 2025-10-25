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
    mu = np.mean(y)
    vy = np.var(y, ddof=0)
    if not np.isfinite(vy) or vy <= 0.0:
        return np.full(d, 1.0 / d, dtype=float)
    base_bins = int(np.clip(np.sqrt(n), 4, 20))
    min_bin = 10
    S = np.zeros(d, dtype=float)
    for j in range(d):
        B = base_bins
        while True:
            q = np.linspace(0.0, 1.0, B + 1)
            edges = np.quantile(x[:, j], q)
            edges[0] = -np.inf
            edges[-1] = np.inf
            idx = np.digitize(x[:, j], edges[1:-1])
            counts = np.bincount(idx, minlength=B)
            if (counts.min() < min_bin) and (B > 2):
                B -= 1
                continue
            break
        sums = np.bincount(idx, weights=y, minlength=B)
        valid = counts >= min_bin
        if np.count_nonzero(valid) < 2:
            S[j] = 0.0
            continue
        mu_b = np.zeros(B, dtype=float)
        mu_b[valid] = sums[valid] / counts[valid]
        p_b = counts[valid] / float(n)
        diff = mu_b[valid] - mu
        S[j] = np.sum(p_b * (diff * diff)) / vy

    S = np.clip(S, 0.0, 1.0)
    total = S.sum()
    if total <= 0.0 or not np.isfinite(total):
        return np.full(d, 1.0 / d, dtype=float)
    return S / total
