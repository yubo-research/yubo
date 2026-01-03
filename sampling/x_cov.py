import numpy as np


def cov_diag(x_0: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    x_0 = np.asarray(x_0)
    assert x.ndim in (1, 2)
    if x.ndim == 1:
        d = x.shape[0]
        x = x.reshape(1, d)
    else:
        d = x.shape[1]
    x_0 = x_0.reshape(-1)
    assert x_0.ndim == 1
    assert x_0.shape[0] == d
    n = x.shape[0]
    assert n >= 1
    diff2 = (x - x_0) ** 2
    v = diff2.mean(axis=0)
    return v.astype(float)


def evec_1(x_0: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    x_0 = np.asarray(x_0)
    assert x.ndim in (1, 2)
    if x.ndim == 1:
        d = x.shape[0]
        x = x.reshape(1, d)
    else:
        d = x.shape[1]
    x_0 = x_0.reshape(-1)
    assert x_0.ndim == 1
    assert x_0.shape[0] == d
    n = x.shape[0]
    assert n >= 1
    centered = x - x_0
    v = np.ones(d, dtype=float)
    v /= np.linalg.norm(v)
    for _ in range(1000):
        y = centered @ v
        w = centered.T @ y
        w /= float(n)
        norm = np.linalg.norm(w)
        if norm == 0.0:
            return w.astype(float)
        w /= norm
        if np.linalg.norm(w - v) <= 1e-12:
            v = w
            break
        v = w
    return v.astype(np.double)
