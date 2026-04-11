"""Accelerated trust-region kernels with clean accel dispatch."""

from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np

from optimizer import trust_region_accel_jax as jax_backend
from optimizer import trust_region_accel_mlx as mlx_backend
from optimizer import trust_region_accel_triton as triton_backend
from optimizer.trust_region_math import (
    _clip_to_unit_box,
    _mahalanobis_sq_from_factor,
    _ray_scale_to_unit_box,
)

ACCEL_MODULES = {
    "mlx": mlx_backend,
    "triton": triton_backend,
    "jax": jax_backend,
}
ACCEL_PRIORITY = ("mlx", "triton", "jax")
ACCEL_ENV = "YUBO_TR_ACCEL"

_ACCEL: str | None = None


def set_accel(name: str) -> None:
    allowed = set(ACCEL_MODULES) | {""}
    if name not in allowed:
        raise ValueError(f"Unknown accel {name!r}, expected one of {allowed}")
    global _ACCEL
    _ACCEL = name


@contextmanager
def accel_override(name: str | None):
    global _ACCEL
    previous = _ACCEL
    try:
        if name is None:
            yield
            return
        if name not in ACCEL_MODULES:
            raise ValueError(f"Unknown accel {name!r}, expected one of {sorted(ACCEL_MODULES)}")
        _ACCEL = name if ACCEL_MODULES[name].available() else ""
        yield
    finally:
        _ACCEL = previous


def _accel_from_env() -> str | None:
    raw = os.environ.get(ACCEL_ENV, "").strip().lower()
    if raw in {"", "auto"}:
        return None
    if raw not in ACCEL_MODULES:
        raise ValueError(f"Unknown {ACCEL_ENV}={raw!r}, expected one of {sorted(ACCEL_MODULES)} or 'auto'")
    if not ACCEL_MODULES[raw].available():
        raise RuntimeError(f"Requested {ACCEL_ENV}={raw!r} but that accel is unavailable")
    return raw


def detect_accel() -> str | None:
    global _ACCEL
    if _ACCEL is not None:
        return _ACCEL if _ACCEL != "" else None
    env_accel = _accel_from_env()
    if env_accel is not None:
        _ACCEL = env_accel
        return env_accel
    for name in ACCEL_PRIORITY:
        if ACCEL_MODULES[name].available():
            _ACCEL = name
            return name
    _ACCEL = ""
    return None


def accel_name(operation: str | None = None) -> str:
    _ = operation
    accel = detect_accel()
    return accel if accel else "none"


def is_available() -> bool:
    return detect_accel() is not None


def current_accel(operation: str | None = None):
    _ = operation
    accel = detect_accel()
    return None if accel is None else ACCEL_MODULES[accel]


class CovCache:
    """Cache accel-specific covariance factorizations by generation."""

    __slots__ = ("chol", "inv", "generation", "accel")

    def __init__(self) -> None:
        self.chol = None
        self.inv = None
        self.generation = -1
        self.accel: str | None = None

    def update(self, cov: np.ndarray, gen: int = -1, *, need_inv: bool = True) -> None:
        accel = accel_name()
        if gen >= 0 and gen == self.generation and accel == self.accel and (not need_inv or self.inv is not None):
            return
        self.accel = accel
        self.generation = gen
        module = current_accel()
        if module is None:
            self.chol = np.linalg.cholesky(cov)
            self.inv = np.linalg.inv(cov) if need_inv else None
            return
        self.chol, self.inv = module.factorize_cov(cov, need_inv=need_inv)

    def invalidate(self) -> None:
        self.chol = None
        self.inv = None
        self.generation = -1
        self.accel = None


def fused_ellipsoid_generate(
    z: np.ndarray,
    x_center: np.ndarray,
    covariance_matrix: np.ndarray,
    length: float,
    cache: CovCache,
    gen: int = -1,
) -> np.ndarray:
    module = current_accel()
    cache.update(covariance_matrix, gen=gen, need_inv=False)
    radius2 = float(length) ** 2
    if module is not None:
        return module.fused_ellipsoid_generate(z, x_center, cache.chol, radius2)
    step = np.asarray(z, dtype=float) @ np.asarray(cache.chol, dtype=float).T
    candidates = _ray_scale_to_unit_box(x_center, np.asarray(x_center, dtype=float).reshape(1, -1) + step)
    delta = candidates - np.asarray(x_center, dtype=float).reshape(1, -1)
    dist2 = _mahalanobis_sq_from_factor(delta, np.asarray(cache.chol, dtype=float))
    bad = np.where(dist2 > radius2 * (1.0 + 1e-8))[0]
    if bad.size > 0:
        scale = np.sqrt(radius2 / np.maximum(dist2[bad], 1e-12))
        delta[bad] *= scale.reshape(-1, 1)
        candidates = _ray_scale_to_unit_box(x_center, np.asarray(x_center, dtype=float).reshape(1, -1) + delta)
    return candidates


def whitened_sample(
    z_tilde: np.ndarray,
    u: np.ndarray,
    length: float,
    radial_mode: str,
    num_dim: int,
) -> np.ndarray | None:
    module = current_accel()
    if module is None:
        return None
    return module.whitened_sample(z_tilde, u, length, radial_mode, num_dim)


def fused_whitened_ellipsoid_candidates(
    z_tilde: np.ndarray,
    u: np.ndarray,
    x_center: np.ndarray,
    covariance_matrix: np.ndarray,
    length: float,
    radial_mode: str,
    num_dim: int,
    cache: CovCache,
    gen: int = -1,
) -> np.ndarray | None:
    module = current_accel()
    cache.update(covariance_matrix, gen=gen, need_inv=False)
    if module is None or not hasattr(module, "fused_whitened_ellipsoid_candidates"):
        return None
    radius2 = float(length) ** 2
    return module.fused_whitened_ellipsoid_candidates(
        z_tilde,
        u,
        x_center,
        cache.chol,
        float(length),
        radial_mode,
        num_dim,
        radius2,
    )


def fused_sobol_ellipsoid_candidates(
    sobol_samples: np.ndarray,
    x_center: np.ndarray,
    covariance_matrix: np.ndarray,
    length: float,
    radial_mode: str,
    num_dim: int,
    prob: float,
    cache: CovCache,
    gen: int = -1,
) -> np.ndarray | None:
    module = current_accel()
    if module is None or not hasattr(module, "fused_sobol_ellipsoid_candidates"):
        return None
    cache.update(covariance_matrix, gen=gen, need_inv=False)
    return module.fused_sobol_ellipsoid_candidates(
        sobol_samples,
        x_center,
        cache.chol,
        float(length),
        radial_mode,
        num_dim,
        float(prob),
        float(length) ** 2,
    )


def fused_metric_candidates(
    z: np.ndarray,
    x_center: np.ndarray,
    cov_factor: np.ndarray,
    length: float,
) -> np.ndarray:
    module = current_accel()
    if module is None:
        step = np.asarray(z, dtype=float) @ np.asarray(cov_factor, dtype=float).T
        return _clip_to_unit_box(np.asarray(x_center, dtype=float), step * float(length))
    return module.fused_metric_candidates(z, x_center, cov_factor, length)


def fused_low_rank_candidates(
    coeff: np.ndarray,
    basis: np.ndarray,
    z: np.ndarray,
    sparse_scale: float,
    x_center: np.ndarray,
    length: float,
) -> np.ndarray:
    module = current_accel()
    if module is None:
        step = np.asarray(coeff, dtype=float) @ np.asarray(basis, dtype=float).T
        if sparse_scale != 0.0:
            step = step + float(sparse_scale) * np.asarray(z, dtype=float)
        return _clip_to_unit_box(np.asarray(x_center, dtype=float), step * float(length))
    return module.fused_low_rank_candidates(coeff, basis, z, sparse_scale, x_center, length)


def mahalanobis_sq(delta: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    module = current_accel()
    if module is None:
        delta_np = np.asarray(delta, dtype=float)
        cov_inv_np = np.asarray(cov_inv, dtype=float)
        return np.einsum("nd,de,ne->n", delta_np, cov_inv_np, delta_np)
    return module.mahalanobis_sq(delta, cov_inv)


def mahalanobis_sq_from_factor(delta: np.ndarray, factor: np.ndarray) -> np.ndarray:
    module = current_accel()
    if module is None:
        delta_np = np.asarray(delta, dtype=float)
        factor_np = np.asarray(factor, dtype=float)
        solved = np.linalg.solve(factor_np, delta_np.T).T
        return np.sum(solved * solved, axis=1)
    return module.mahalanobis_sq_from_factor(delta, factor)


def mahalanobis_sq_from_cov(delta: np.ndarray, cov: np.ndarray) -> np.ndarray:
    module = current_accel()
    if module is None:
        delta_np = np.asarray(delta, dtype=float)
        cov_np = np.asarray(cov, dtype=float)
        solved = np.linalg.solve(cov_np, delta_np.T).T
        return np.sum(delta_np * solved, axis=1)
    return module.mahalanobis_sq_from_cov(delta, cov)


def low_rank_step(coeff: np.ndarray, basis: np.ndarray) -> np.ndarray:
    module = current_accel()
    if module is None:
        return np.asarray(coeff, dtype=float) @ np.asarray(basis, dtype=float).T
    return module.low_rank_step(coeff, basis)


def low_rank_step_with_sparse(
    coeff: np.ndarray,
    basis: np.ndarray,
    z: np.ndarray,
    scale: float,
) -> np.ndarray:
    module = current_accel()
    if module is None:
        step = np.asarray(coeff, dtype=float) @ np.asarray(basis, dtype=float).T
        if scale != 0.0:
            step = step + float(scale) * np.asarray(z, dtype=float)
        return step
    return module.low_rank_step_with_sparse(coeff, basis, z, scale)


def low_rank_metric(
    delta: np.ndarray,
    basis: np.ndarray,
    beta: np.ndarray,
    inv_alpha: float,
) -> np.ndarray:
    module = current_accel()
    if module is None:
        delta_np = np.asarray(delta, dtype=float)
        basis_np = np.asarray(basis, dtype=float)
        beta_np = np.asarray(beta, dtype=float)
        proj = delta_np @ basis_np
        return inv_alpha * np.sum(delta_np * delta_np, axis=1) - np.sum(proj * proj * beta_np, axis=1)
    return module.low_rank_metric(delta, basis, beta, inv_alpha)


def clip_to_unit_box(x_center: np.ndarray, step: np.ndarray) -> np.ndarray:
    module = current_accel()
    if module is None:
        return _clip_to_unit_box(x_center, step)
    return module.clip_to_unit_box(x_center, step)


def ray_scale_to_unit_box(x_center: np.ndarray, x: np.ndarray) -> np.ndarray:
    center = np.asarray(x_center, dtype=float)
    return clip_to_unit_box(center, np.asarray(x, dtype=float) - center)


def cholesky(cov: np.ndarray) -> np.ndarray:
    module = current_accel()
    if module is None:
        return np.linalg.cholesky(np.asarray(cov, dtype=float))
    return module.cholesky(cov)


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    module = current_accel()
    if module is None:
        return np.asarray(a, dtype=float) @ np.asarray(b, dtype=float)
    return module.matmul(a, b)
