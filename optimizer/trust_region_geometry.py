"""
Geometry models for shaped trust regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

import optimizer.trust_region_accel as _accel
from optimizer.trust_region_math import (
    _ensure_spd,
    _full_factor,
    _low_rank_factor,
    _low_rank_factor_from_cov,
    _LowRankFactor,
    _mahalanobis_sq_from_inv,
    _normalize_weights,
    _ray_scale_to_unit_box,
    _symmetric_spd_factor,
    _trace_normalize,
)
from optimizer.trust_region_sampling_utils import (
    _full_factor_from_direction,
    _low_rank_factor_from_direction,
    _prepare_gradient_geometry_inputs,
)

CovmatKind = Literal["dense", "low_rank"]


@dataclass
class _MetricGeometryModel:
    num_dim: int
    covmat: CovmatKind
    metric_rank: int | None
    use_accel: bool = False
    cov_factor: np.ndarray = field(init=False)
    low_rank: _LowRankFactor = field(init=False)
    has_geometry: bool = field(default=False, init=False)
    _cov_gen: int = field(default=0, init=False, repr=False)
    _cached_cov: np.ndarray | None = field(default=None, init=False, repr=False)
    _cached_cov_inv: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.cov_factor = np.eye(self.num_dim, dtype=float)
        self.low_rank = _LowRankFactor(
            sqrt_alpha=1.0,
            basis=np.zeros((self.num_dim, 0), dtype=float),
            sqrt_vals=np.zeros(0),
        )
        self.has_geometry = False
        self._cov_gen += 1
        self._cached_cov = None
        self._cached_cov_inv = None

    def set_geometry(self, delta_x: np.ndarray | Any, weights: np.ndarray | Any) -> None:
        dx = np.asarray(delta_x, dtype=float)
        if dx.ndim != 2 or dx.shape[0] == 0:
            return
        if dx.shape[1] != self.num_dim:
            raise ValueError(f"delta_x has incompatible shape {dx.shape} for num_dim={self.num_dim}")
        w = _normalize_weights(np.asarray(weights, dtype=float))
        if w is None or w.shape[0] != dx.shape[0]:
            return
        mean = np.sum(w[:, None] * dx, axis=0)
        centered = dx - mean
        cov = None
        if self.covmat == "dense":
            cov = centered.T @ (w[:, None] * centered)
            cov = _trace_normalize(cov, self.num_dim)
        self.update_from_cov(centered=centered, weights=w, cov=cov)

    def set_gradient_geometry(
        self,
        delta_x: np.ndarray | Any,
        delta_y: np.ndarray | Any,
        weights: np.ndarray | Any,
        *,
        eps_norm: float = 1e-12,
    ) -> None:
        prepared = _prepare_gradient_geometry_inputs(
            delta_x=delta_x,
            delta_y=delta_y,
            weights=weights,
            num_dim=self.num_dim,
            eps_norm=eps_norm,
        )
        if prepared is None:
            return
        scaled_dx, w = prepared
        cov = None
        if self.covmat == "dense":
            weighted = scaled_dx * np.sqrt(w).reshape(-1, 1)
            cov = weighted.T @ weighted
            cov = _trace_normalize(cov, self.num_dim)
        self.update_from_cov(centered=scaled_dx, weights=w, cov=cov)

    def set_analytic_gradient(self, grad: np.ndarray | Any) -> None:
        g = np.asarray(grad, dtype=float).reshape(-1)
        if g.shape[0] != self.num_dim or not np.any(np.isfinite(g)):
            return
        g_norm = float(np.linalg.norm(g))
        if g_norm <= 0.0:
            return
        unit = g / g_norm
        self._cov_gen += 1
        self._cached_cov = None
        self._cached_cov_inv = None
        self.has_geometry = True
        if self.covmat == "dense":
            self.cov_factor = _full_factor_from_direction(
                unit,
                dim=self.num_dim,
                lam_min=1e-4,
                eps=1e-6,
            )
            return
        if self.covmat != "low_rank":
            raise ValueError(f"Unknown covmat: {self.covmat!r}")
        self.low_rank = _low_rank_factor_from_direction(
            unit,
            dim=self.num_dim,
            eps=1e-6,
        )
        self.cov_factor = self.low_rank.basis * self.low_rank.sqrt_vals.reshape(1, -1)

    def update_from_cov(
        self,
        *,
        centered: np.ndarray,
        weights: np.ndarray,
        cov: np.ndarray | None,
    ) -> None:
        _ = centered, weights
        self._cov_gen += 1
        self._cached_cov = None
        self._cached_cov_inv = None
        lam_min = 1e-4
        lam_max = 1e4
        eps = 1e-6
        if self.covmat == "dense":
            if cov is None:
                raise ValueError("cov is required for covmat='dense'")
            factor = _full_factor(cov, dim=self.num_dim, lam_min=lam_min, lam_max=lam_max, eps=eps)
            if factor is None:
                return
            self.cov_factor = factor
            self.has_geometry = True
            return
        if self.covmat != "low_rank":
            raise ValueError(f"Unknown covmat: {self.covmat!r}")
        rank_cap = int(self.metric_rank) if self.metric_rank is not None else None
        low_rank = None
        if cov is not None:
            low_rank = _low_rank_factor_from_cov(
                cov,
                dim=self.num_dim,
                lam_min=lam_min,
                lam_max=lam_max,
                eps=eps,
                rank_cap=rank_cap,
            )
        if low_rank is None:
            low_rank = _low_rank_factor(
                centered,
                weights,
                dim=self.num_dim,
                lam_min=lam_min,
                lam_max=lam_max,
                eps=eps,
                rank_cap=rank_cap,
            )
        if low_rank is None:
            return
        self.low_rank = low_rank
        self.cov_factor = low_rank.basis * low_rank.sqrt_vals.reshape(1, -1)
        self.has_geometry = True

    def build_step(self, z: np.ndarray, rng: Any) -> np.ndarray:
        if self.covmat == "dense":
            z_arr = np.asarray(z, dtype=float)
            if self.use_accel:
                return _accel.matmul(z_arr, self.cov_factor.T)
            return z_arr @ np.asarray(self.cov_factor, dtype=float).T
        if self.covmat != "low_rank":
            raise ValueError(self.covmat)
        z_arr = np.asarray(z, dtype=float)
        basis = np.asarray(self.low_rank.basis, dtype=float)
        sqrt_vals = np.asarray(self.low_rank.sqrt_vals, dtype=float)
        if basis.ndim != 2 or basis.shape[0] != self.num_dim:
            raise RuntimeError(basis.shape)
        if sqrt_vals.ndim != 1 or basis.shape[1] != sqrt_vals.shape[0]:
            raise RuntimeError((basis.shape, sqrt_vals.shape))
        rank = int(sqrt_vals.shape[0])
        if rank == 0:
            return z_arr * float(self.low_rank.sqrt_alpha)
        coeff = rng.uniform(-0.5, 0.5, size=(z.shape[0], rank)) * sqrt_vals.reshape(1, -1)
        if self.use_accel:
            return _accel.low_rank_step_with_sparse(
                coeff,
                basis,
                z_arr,
                float(self.low_rank.sqrt_alpha),
            )
        step = coeff @ basis.T
        sparse_scale = float(self.low_rank.sqrt_alpha)
        if sparse_scale != 0.0:
            step = step + sparse_scale * z_arr
        return step

    def build_candidates(
        self,
        z: np.ndarray,
        rng: Any,
        x_center: np.ndarray,
        length: float,
    ) -> np.ndarray:
        if self.covmat == "dense":
            step = self.build_step(z, rng) * float(length)
            x_center_arr = np.asarray(x_center, dtype=float)
            if self.use_accel:
                return _accel.clip_to_unit_box(x_center_arr, step)
            return _ray_scale_to_unit_box(x_center_arr, x_center_arr.reshape(1, -1) + step)
        if self.covmat != "low_rank":
            raise ValueError(self.covmat)
        z_arr = np.asarray(z, dtype=float)
        x_center_arr = np.asarray(x_center, dtype=float)
        basis = np.asarray(self.low_rank.basis, dtype=float)
        sqrt_vals = np.asarray(self.low_rank.sqrt_vals, dtype=float)
        if basis.ndim != 2 or basis.shape[0] != self.num_dim:
            raise RuntimeError(basis.shape)
        if sqrt_vals.ndim != 1 or basis.shape[1] != sqrt_vals.shape[0]:
            raise RuntimeError((basis.shape, sqrt_vals.shape))
        rank = int(sqrt_vals.shape[0])
        sparse_scale = float(self.low_rank.sqrt_alpha)
        if rank == 0:
            step = z_arr * sparse_scale * float(length)
            if self.use_accel:
                return _accel.clip_to_unit_box(x_center_arr, step)
            return _ray_scale_to_unit_box(x_center_arr, x_center_arr.reshape(1, -1) + step)
        coeff = rng.uniform(-0.5, 0.5, size=(z.shape[0], rank)) * sqrt_vals.reshape(1, -1)
        if self.use_accel:
            return _accel.fused_low_rank_candidates(
                coeff,
                basis,
                z_arr,
                sparse_scale,
                x_center_arr,
                float(length),
            )
        step = coeff @ basis.T
        if sparse_scale != 0.0:
            step = step + sparse_scale * z_arr
        step = step * float(length)
        return _ray_scale_to_unit_box(x_center_arr, x_center_arr.reshape(1, -1) + step)

    def covariance_matrix(self, *, jitter: float) -> np.ndarray:
        if self._cached_cov is not None:
            return self._cached_cov
        if self.covmat == "dense":
            cov = self.cov_factor @ self.cov_factor.T
            cov = 0.5 * (cov + cov.T)
            cov += jitter * max(1.0, float(np.max(np.abs(cov)))) * np.eye(cov.shape[0])
        else:
            low_rank = self.low_rank
            alpha = float(low_rank.sqrt_alpha) ** 2
            basis = np.asarray(low_rank.basis, dtype=float)
            lam = np.asarray(low_rank.sqrt_vals, dtype=float) ** 2
            cov = alpha * np.eye(self.num_dim, dtype=float)
            if basis.size > 0 and lam.size > 0:
                cov = cov + (basis * lam.reshape(1, -1)) @ basis.T
            cov = _ensure_spd(cov, jitter=jitter)
        self._cached_cov = cov
        self._cached_cov_inv = None
        return cov

    def covariance_inverse(self, *, jitter: float) -> np.ndarray:
        if self._cached_cov_inv is not None:
            return self._cached_cov_inv
        self._cached_cov_inv = np.linalg.inv(self.covariance_matrix(jitter=jitter))
        return self._cached_cov_inv

    def restricted_covariance(self, indices: np.ndarray, *, jitter: float) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idx.size <= 0:
            raise ValueError("indices must be non-empty")
        metric = self.covariance_inverse(jitter=jitter)
        metric_sub = _ensure_spd(np.asarray(metric[np.ix_(idx, idx)], dtype=float), jitter=jitter)
        cov_sub = np.linalg.inv(metric_sub)
        return _ensure_spd(cov_sub, jitter=jitter)

    def restricted_factor(self, indices: np.ndarray, *, jitter: float) -> np.ndarray:
        return _symmetric_spd_factor(self.restricted_covariance(indices, jitter=jitter), jitter=jitter)

    def mahalanobis_sq(self, delta: np.ndarray, *, jitter: float) -> np.ndarray:
        if self.covmat == "low_rank":
            basis = np.asarray(self.low_rank.basis, dtype=float)
            alpha = max(float(self.low_rank.sqrt_alpha) ** 2, 1e-12)
            inv_alpha = 1.0 / alpha
            lam = np.asarray(self.low_rank.sqrt_vals, dtype=float) ** 2
            beta = inv_alpha * lam / (alpha + lam)
            if self.use_accel:
                return _accel.low_rank_metric(delta, basis, beta, inv_alpha)
            with _accel.accel_override(""):
                return _accel.low_rank_metric(delta, basis, beta, inv_alpha)
        return _mahalanobis_sq_from_inv(delta, self.covariance_inverse(jitter=jitter))


@dataclass
class _TrueEllipsoidGeometryModel(_MetricGeometryModel):
    update_option: Literal["option_a", "option_b", "option_c"] = "option_a"
    shape_period: int = 5
    shape_ema: float = 0.2
    shape_jitter: float = 1e-6
    shape_kappa_max: float = 1e4
    shape_tick: int = field(default=0, init=False)
    ema_cov: np.ndarray | None = field(default=None, init=False)

    def reset(self) -> None:
        super().reset()
        self.shape_tick = 0
        self.ema_cov = None

    def _condition_shape_covariance(self, cov: np.ndarray) -> np.ndarray:
        mat = _trace_normalize(np.asarray(cov, dtype=float), self.num_dim)
        mat = _ensure_spd(mat, jitter=float(self.shape_jitter))
        if self.update_option != "option_b":
            return mat
        eigvals, eigvecs = np.linalg.eigh(mat)
        eigvals = np.maximum(eigvals, 1e-12)
        eig_max = float(np.max(eigvals))
        if not np.isfinite(eig_max) or eig_max <= 0.0:
            return mat
        eig_min = max(eig_max / float(self.shape_kappa_max), 1e-12)
        eigvals = np.clip(eigvals, eig_min, eig_max)
        mat = (eigvecs * eigvals.reshape(1, -1)) @ eigvecs.T
        return _trace_normalize(mat, self.num_dim)

    def set_geometry(self, delta_x: np.ndarray | Any, weights: np.ndarray | Any) -> None:
        super().set_geometry(delta_x=delta_x, weights=weights)
        if self.update_option in ("option_a", "option_b"):
            self._apply_shape_update(np.asarray(delta_x, dtype=float), np.asarray(weights, dtype=float))

    def _apply_shape_update(self, delta_x: np.ndarray, weights: np.ndarray) -> None:
        dx = np.asarray(delta_x, dtype=float)
        if dx.ndim != 2 or dx.shape[0] == 0:
            return
        w = _normalize_weights(np.asarray(weights, dtype=float))
        if w is None or w.shape[0] != dx.shape[0]:
            return
        self.shape_tick += 1
        if self.shape_tick % int(self.shape_period) != 0:
            return
        mean = np.sum(w[:, None] * dx, axis=0)
        centered = dx - mean
        cov_hat = centered.T @ (w[:, None] * centered)
        cov_hat = self._condition_shape_covariance(cov_hat)
        eta = float(self.shape_ema)
        if self.ema_cov is None:
            self.ema_cov = cov_hat
        else:
            self.ema_cov = _ensure_spd(
                (1.0 - eta) * self.ema_cov + eta * cov_hat,
                jitter=float(self.shape_jitter),
            )
        self.ema_cov = self._condition_shape_covariance(self.ema_cov)
        self.update_from_cov(centered=centered, weights=w, cov=self.ema_cov)
