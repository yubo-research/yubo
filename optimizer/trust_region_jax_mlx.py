"""MLX backend helpers for trust-region accelerated kernels."""

from __future__ import annotations

import numpy as np


def _mlx():
    import mlx.core as mx

    return mx


def _mlx_mahalanobis_sq(delta: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    mx = _mlx()
    d = mx.array(delta.astype(np.float32))
    ci = mx.array(cov_inv.astype(np.float32))
    out = mx.sum(d * (d @ ci), axis=1)
    mx.eval(out)
    return np.asarray(out, dtype=np.float64)


def _mlx_mahalanobis_sq_from_cov(delta: np.ndarray, cov: np.ndarray) -> np.ndarray:
    mx = _mlx()
    d = mx.array(delta.astype(np.float32))
    c = mx.array(cov.astype(np.float32))
    ci = mx.linalg.inv(c, stream=mx.cpu)
    out = mx.sum(d * (d @ ci), axis=1)
    mx.eval(out)
    return np.asarray(out, dtype=np.float64)


def _mlx_low_rank_step(coeff: np.ndarray, basis: np.ndarray) -> np.ndarray:
    mx = _mlx()
    c = mx.array(coeff.astype(np.float32))
    b = mx.array(basis.astype(np.float32))
    out = c @ b.T
    mx.eval(out)
    return np.asarray(out, dtype=np.float64)


def _mlx_low_rank_metric(
    delta: np.ndarray,
    basis: np.ndarray,
    beta: np.ndarray,
    inv_alpha: float,
) -> np.ndarray:
    mx = _mlx()
    d = mx.array(delta.astype(np.float32))
    b = mx.array(basis.astype(np.float32))
    be = mx.array(beta.astype(np.float32))
    proj = d @ b
    term1 = mx.array(np.float32(inv_alpha)) * mx.sum(d * d, axis=1)
    term2 = mx.sum(proj * proj * be, axis=1)
    out = term1 - term2
    mx.eval(out)
    return np.asarray(out, dtype=np.float64)


def _mlx_clip_step(xc, s):
    """Vectorized clip: xc + s * t_min, result in [0,1]^D. Operates on MLX arrays."""
    mx = _mlx()
    safe = mx.where(s == 0, 1.0, s)
    tp = (1.0 - xc) / safe
    tn = -xc / safe
    t_all = mx.where(s > 0, tp, mx.where(s < 0, tn, mx.array(1e30)))
    t_min = mx.clip(mx.min(t_all, axis=1), 0.0, 1.0)
    return xc + s * t_min[:, None]


def _mlx_clip_to_unit_box(x_center: np.ndarray, step: np.ndarray) -> np.ndarray:
    mx = _mlx()
    xc = mx.array(x_center.astype(np.float32)).reshape(1, -1)
    s = mx.array(step.astype(np.float32))
    out = _mlx_clip_step(xc, s)
    mx.eval(out)
    return np.asarray(out, dtype=np.float64)


def _mlx_cholesky(cov: np.ndarray) -> np.ndarray:
    mx = _mlx()
    c = mx.array(cov.astype(np.float32))
    try:
        out = mx.linalg.cholesky(c)
    except ValueError:
        out = mx.linalg.cholesky(c, stream=mx.cpu)
    mx.eval(out)
    return np.asarray(out, dtype=np.float64)


def _mlx_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mx = _mlx()
    ma = mx.array(a.astype(np.float32))
    mb = mx.array(b.astype(np.float32))
    out = ma @ mb
    mx.eval(out)
    return np.asarray(out, dtype=np.float64)
