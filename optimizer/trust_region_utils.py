"""
References:
- TuRBO: Eriksson et al., NeurIPS 2019 (Scalable Global Optimization via Local Bayesian Optimization),
  https://arxiv.org/abs/1910.01739
- Classical trust-region ratio update: Conn, Gould, Toint, *Trust Region Methods* (SIAM, 2000),
  https://doi.org/10.1137/1.9780898719857
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
from enn.turbo.config.candidate_rv import CandidateRV
from enn.turbo.turbo_utils import generate_raasp_candidates

import optimizer.trust_region_accel as _accel
from optimizer.pc_rotation import PCRotationMode, PCRotationResult
from optimizer.trust_region_math import (
    _add_sparse_axis,
    _apply_full_factor,
    _clip_to_unit_box,
    _ensure_spd,
    _full_factor,
    _low_rank_factor,
    _low_rank_factor_from_cov,
    _LowRankFactor,
    _mahalanobis_sq,
    _mahalanobis_sq_from_factor,
    _mahalanobis_sq_from_inv,
    _normalize_weights,
    _ray_scale_to_unit_box,
    _trace_normalize,
)
from optimizer.trust_region_sampling_utils import (
    _candidate_rv_name,
    _draw_sobol_prefix,
    _full_factor_from_direction,
    _generate_raasp_candidates_fast_sobol,
    _generate_raasp_candidates_fast_uniform,
    _low_rank_factor_from_direction,
    _low_rank_mahalanobis_sq,
    _low_rank_symmetric_sqrt_step,
    _prepare_gradient_geometry_inputs,
    _sample_whitened_inputs,
    _whitened_sample_numpy,
)

SamplerKind = Literal["full", "low_rank"]
RadialMode = Literal["ball_uniform", "boundary"]
UpdateMode = Literal["option_a", "option_b", "option_c"]


@dataclass
class _MetricGeometryModel:
    num_dim: int
    metric_sampler: SamplerKind
    metric_rank: int | None
    pc_rotation_mode: PCRotationMode | None = None
    pc_rank: int | None = None
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
        if self.metric_sampler == "full":
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
        if self.metric_sampler == "full":
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
        if self.metric_sampler == "full":
            self.cov_factor = _full_factor_from_direction(
                unit,
                dim=self.num_dim,
                lam_min=1e-4,
                eps=1e-6,
            )
            return
        if self.metric_sampler != "low_rank":
            raise ValueError(f"Unknown metric_sampler: {self.metric_sampler!r}")
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
        if self.metric_sampler == "full":
            if cov is None:
                raise ValueError("cov is required for metric_sampler='full'")
            factor = _full_factor(cov, dim=self.num_dim, lam_min=lam_min, lam_max=lam_max, eps=eps)
            if factor is None:
                return
            self.cov_factor = factor
            self.has_geometry = True
            return
        if self.metric_sampler != "low_rank":
            raise ValueError(f"Unknown metric_sampler: {self.metric_sampler!r}")
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

    def update_from_pc_rotation(
        self,
        rotation_result: PCRotationResult,
        *,
        trace_scale: float = 1.0,
        null_space_alpha: float = 1e-4,
    ) -> None:
        """Update geometry from LABCAT PC rotation (see pc_rotation.LABCAT_CITATION).

        Uses the principal directions and singular values to build an ellipsoidal
        trust region aligned with the weighted PCs. Full mode uses all PCs;
        low_rank mode uses top-k PCs with isotropic residual for the null space.
        """
        if not rotation_result.has_rotation:
            return
        self._cov_gen += 1
        self._cached_cov = None
        self._cached_cov_inv = None
        basis = np.asarray(rotation_result.basis, dtype=float)
        s = np.asarray(rotation_result.singular_values, dtype=float)
        if basis.shape[0] != self.num_dim or s.size != basis.shape[1]:
            return
        s = np.maximum(s, 1e-10)
        rank = int(basis.shape[1])
        dim = self.num_dim

        use_full = self.pc_rotation_mode == "full" and rank >= dim
        if use_full:
            # Full: all d PCs, cov = V diag(s^2) V^T
            factor = basis * s.reshape(1, -1)
            cov = factor @ factor.T
            cov = _trace_normalize(cov, dim)
            cov = _ensure_spd(cov, jitter=1e-8)
            lam_min, lam_max, eps = 1e-4, 1e4, 1e-6
            full_f = _full_factor(cov, dim=dim, lam_min=lam_min, lam_max=lam_max, eps=eps)
            if full_f is not None:
                self.cov_factor = full_f
                self.low_rank = _LowRankFactor(
                    sqrt_alpha=0.0,
                    basis=np.zeros((dim, 0), dtype=float),
                    sqrt_vals=np.zeros(0),
                )
                self.metric_sampler = "full"
                self.has_geometry = True
            return

        # Full with rank < d, or low_rank: PCs + isotropic null space
        total = float(np.sum(s**2))
        if not np.isfinite(total) or total <= 0.0:
            return
        scale = trace_scale * float(dim) / (total + null_space_alpha * max(0, dim - rank))
        sqrt_vals = np.sqrt(s**2 * scale)
        sqrt_vals = np.clip(sqrt_vals, 1e-4, 1e4)
        sqrt_alpha = float(np.sqrt(null_space_alpha * scale))
        self.low_rank = _LowRankFactor(
            sqrt_alpha=sqrt_alpha,
            basis=basis,
            sqrt_vals=sqrt_vals,
        )
        self.cov_factor = basis * sqrt_vals.reshape(1, -1)
        self.metric_sampler = "low_rank"
        self.has_geometry = True

    def build_step(self, z: np.ndarray, rng: Any) -> np.ndarray:
        if self.metric_sampler == "full":
            if self.use_accel:
                return _accel.matmul(z, self.cov_factor.T)
            return _apply_full_factor(z, self.cov_factor)
        if self.metric_sampler != "low_rank":
            raise ValueError(self.metric_sampler)
        basis = np.asarray(self.low_rank.basis, dtype=float)
        sqrt_vals = np.asarray(self.low_rank.sqrt_vals, dtype=float)
        if basis.ndim != 2 or basis.shape[0] != self.num_dim:
            raise RuntimeError(basis.shape)
        if sqrt_vals.ndim != 1 or basis.shape[1] != sqrt_vals.shape[0]:
            raise RuntimeError((basis.shape, sqrt_vals.shape))
        rank = int(sqrt_vals.shape[0])
        if rank == 0:
            step = np.asarray(z, dtype=float) * float(self.low_rank.sqrt_alpha)
        else:
            coeff = rng.uniform(-0.5, 0.5, size=(z.shape[0], rank)) * sqrt_vals.reshape(1, -1)
            if self.use_accel:
                step = _accel.low_rank_step_with_sparse(
                    coeff,
                    basis,
                    z,
                    float(self.low_rank.sqrt_alpha),
                )
            else:
                step = coeff @ basis.T
                _add_sparse_axis(step, z, float(self.low_rank.sqrt_alpha))
        return step

    def build_candidates(
        self,
        z: np.ndarray,
        rng: Any,
        x_center: np.ndarray,
        length: float,
    ) -> np.ndarray:
        if self.metric_sampler == "full":
            step = self.build_step(z, rng) * float(length)
            if self.use_accel:
                return _accel.clip_to_unit_box(np.asarray(x_center, dtype=float), step)
            return _clip_to_unit_box(np.asarray(x_center, dtype=float), step)
        if self.metric_sampler != "low_rank":
            raise ValueError(self.metric_sampler)
        basis = np.asarray(self.low_rank.basis, dtype=float)
        sqrt_vals = np.asarray(self.low_rank.sqrt_vals, dtype=float)
        if basis.ndim != 2 or basis.shape[0] != self.num_dim:
            raise RuntimeError(basis.shape)
        if sqrt_vals.ndim != 1 or basis.shape[1] != sqrt_vals.shape[0]:
            raise RuntimeError((basis.shape, sqrt_vals.shape))
        rank = int(sqrt_vals.shape[0])
        sparse_scale = float(self.low_rank.sqrt_alpha)
        if rank == 0:
            step = np.asarray(z, dtype=float) * sparse_scale * float(length)
            if self.use_accel:
                return _accel.clip_to_unit_box(np.asarray(x_center, dtype=float), step)
            return _clip_to_unit_box(np.asarray(x_center, dtype=float), step)
        coeff = rng.uniform(-0.5, 0.5, size=(z.shape[0], rank)) * sqrt_vals.reshape(1, -1)
        if self.use_accel:
            return _accel.fused_low_rank_candidates(
                coeff,
                basis,
                z,
                sparse_scale,
                np.asarray(x_center, dtype=float),
                float(length),
            )
        step = coeff @ basis.T
        _add_sparse_axis(step, z, sparse_scale)
        return _clip_to_unit_box(np.asarray(x_center, dtype=float), step * float(length))

    def covariance_matrix(self, *, jitter: float) -> np.ndarray:
        if self._cached_cov is not None:
            return self._cached_cov
        if self.metric_sampler == "full":
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

    def mahalanobis_sq(self, delta: np.ndarray, *, jitter: float) -> np.ndarray:
        if self.metric_sampler == "low_rank":
            return _low_rank_mahalanobis_sq(delta, self.low_rank, use_accel=self.use_accel)
        return _mahalanobis_sq_from_inv(delta, self.covariance_inverse(jitter=jitter))


@dataclass
class _TrueEllipsoidGeometryModel(_MetricGeometryModel):
    update_option: UpdateMode = "option_a"
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
        """Apply repository-specific shape-update heuristics for option_a/option_b.

        Note: this update controller is an in-repo heuristic layer, not a direct
        implementation of a single published algorithm.
        """
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


@dataclass(frozen=True)
class _AxisAlignedStepSampler:
    default_candidate_rv: CandidateRV
    use_accel: bool = False

    def generate(
        self,
        *,
        x_center: np.ndarray,
        length: float,
        num_dim: int,
        num_candidates: int,
        rng: Any,
        candidate_rv: CandidateRV | None,
        sobol_engine: Any | None,
        num_pert: int,
        build_step: Callable[[np.ndarray, Any], np.ndarray],
        build_candidates: Callable[[np.ndarray, Any, np.ndarray, float], np.ndarray] | None = None,
        cov_factor: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate TuRBO-style RAASP candidates, then apply geometry transform.

        Candidate perturbations are produced by ENN's `generate_raasp_candidates`
        helper (TuRBO lineage), then mapped through the current geometry model.
        """
        z_center = np.zeros(num_dim, dtype=float)
        lb = -0.5 * np.ones(num_dim, dtype=float)
        ub = 0.5 * np.ones(num_dim, dtype=float)
        rv = self.default_candidate_rv if candidate_rv is None else candidate_rv
        rv_name = _candidate_rv_name(rv)
        if self.use_accel and rv_name in {"uniform", "gpu_uniform"}:
            z = _generate_raasp_candidates_fast_uniform(
                z_center,
                lb,
                ub,
                num_candidates,
                rng=rng,
                num_pert=num_pert,
            )
        elif self.use_accel and rv_name == "sobol":
            if sobol_engine is None:
                raise ValueError("sobol_engine required for CandidateRV.SOBOL")
            z = _generate_raasp_candidates_fast_sobol(
                z_center,
                lb,
                ub,
                num_candidates,
                rng=rng,
                sobol_engine=sobol_engine,
                num_pert=num_pert,
            )
        else:
            z = generate_raasp_candidates(
                z_center,
                lb,
                ub,
                num_candidates,
                rng=rng,
                candidate_rv=rv,
                sobol_engine=sobol_engine,
                num_pert=num_pert,
            )
        if self.use_accel and cov_factor is not None:
            return _accel.fused_metric_candidates(
                z,
                np.asarray(x_center, dtype=float),
                cov_factor,
                float(length),
            )
        if self.use_accel and build_candidates is not None:
            return build_candidates(
                z,
                rng,
                np.asarray(x_center, dtype=float),
                float(length),
            )
        step = build_step(z, rng) * float(length)
        if self.use_accel:
            return _accel.clip_to_unit_box(np.asarray(x_center, dtype=float), step)
        return _clip_to_unit_box(np.asarray(x_center, dtype=float), step)


@dataclass
class _TrueEllipsoidStepSampler:
    default_candidate_rv: CandidateRV
    p_raasp: float
    radial_mode: RadialMode
    use_accel: bool = False
    _cov_cache: _accel.CovCache = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_cov_cache", _accel.CovCache())

    def _sample_raasp_whitened(
        self,
        *,
        num_candidates: int,
        num_dim: int,
        length: float,
        rng: Any,
        candidate_rv: CandidateRV,
        sobol_engine: Any | None,
    ) -> np.ndarray:
        """Sample directions in whitened space, then draw a radius.

        `ball_uniform` uses rho = u^(1/d), which yields uniform volume density
        inside a d-ball. `boundary` is an in-repo boundary-focused heuristic.
        """
        prob = float(self.p_raasp)
        z_tilde, u = _sample_whitened_inputs(
            num_candidates=num_candidates,
            num_dim=num_dim,
            rng=rng,
            candidate_rv=candidate_rv,
            prob=prob,
            sobol_engine=sobol_engine,
        )
        if self.use_accel and num_candidates * num_dim >= 250_000:
            result = _accel.whitened_sample(
                z_tilde,
                u,
                float(length),
                self.radial_mode,
                num_dim,
            )
            if result is not None:
                return result
        return _whitened_sample_numpy(
            z_tilde=z_tilde,
            u=u,
            length=length,
            radial_mode=self.radial_mode,
            num_dim=num_dim,
            rng=rng,
        )

    def _generate_low_rank(
        self,
        *,
        x_center: np.ndarray,
        num_dim: int,
        num_candidates: int,
        length: float,
        rng: Any,
        candidate_rv: CandidateRV,
        sobol_engine: Any | None,
        low_rank: _LowRankFactor,
    ) -> np.ndarray:
        z = self._sample_raasp_whitened(
            num_candidates=num_candidates,
            num_dim=num_dim,
            length=length,
            rng=rng,
            candidate_rv=candidate_rv,
            sobol_engine=sobol_engine,
        )
        step = _low_rank_symmetric_sqrt_step(z, low_rank, use_accel=self.use_accel)
        candidates = _ray_scale_to_unit_box(x_center, x_center.reshape(1, -1) + step)
        delta = candidates - x_center.reshape(1, -1)
        dist2 = _low_rank_mahalanobis_sq(delta, low_rank, use_accel=self.use_accel)
        radius2 = float(length) ** 2
        bad = np.where(dist2 > radius2 * (1.0 + 1e-8))[0]
        if bad.size > 0:
            scale = np.sqrt(radius2 / np.maximum(dist2[bad], 1e-12))
            delta[bad] *= scale.reshape(-1, 1)
            candidates = _ray_scale_to_unit_box(x_center, x_center.reshape(1, -1) + delta)
        return candidates

    def _generate_cov_factor(
        self,
        *,
        x_center: np.ndarray,
        num_dim: int,
        num_candidates: int,
        length: float,
        rng: Any,
        candidate_rv: CandidateRV,
        sobol_engine: Any | None,
        cov_factor: np.ndarray,
    ) -> np.ndarray:
        z = self._sample_raasp_whitened(
            num_candidates=num_candidates,
            num_dim=num_dim,
            length=length,
            rng=rng,
            candidate_rv=candidate_rv,
            sobol_engine=sobol_engine,
        )
        step = z @ np.asarray(cov_factor, dtype=float).T
        candidates = _ray_scale_to_unit_box(x_center, x_center.reshape(1, -1) + step)
        delta = candidates - x_center.reshape(1, -1)
        dist2 = _mahalanobis_sq_from_factor(delta, cov_factor)
        radius2 = float(length) ** 2
        bad = np.where(dist2 > radius2 * (1.0 + 1e-8))[0]
        if bad.size > 0:
            scale = np.sqrt(radius2 / np.maximum(dist2[bad], 1e-12))
            delta[bad] *= scale.reshape(-1, 1)
            candidates = _ray_scale_to_unit_box(x_center, x_center.reshape(1, -1) + delta)
        return candidates

    def _generate_with_jax(
        self,
        *,
        x_center: np.ndarray,
        num_dim: int,
        num_candidates: int,
        length: float,
        rng: Any,
        candidate_rv: CandidateRV,
        sobol_engine: Any | None,
        covariance_matrix: np.ndarray,
        cov_gen: int,
    ) -> np.ndarray | None:
        rv_name = _candidate_rv_name(candidate_rv)
        if rv_name == "sobol":
            if sobol_engine is None:
                raise ValueError("sobol_engine required for CandidateRV.SOBOL")
            sobol_samples = np.asarray(_draw_sobol_prefix(sobol_engine, int(num_candidates)), dtype=np.float32)
            candidates = _accel.fused_sobol_ellipsoid_candidates(
                sobol_samples,
                x_center,
                covariance_matrix,
                float(length),
                self.radial_mode,
                num_dim,
                float(self.p_raasp),
                self._cov_cache,
                gen=cov_gen,
            )
            if candidates is not None:
                return candidates
        z_tilde, u = _sample_whitened_inputs(
            num_candidates=num_candidates,
            num_dim=num_dim,
            rng=rng,
            candidate_rv=candidate_rv,
            prob=float(self.p_raasp),
            sobol_engine=sobol_engine,
        )
        candidates = _accel.fused_whitened_ellipsoid_candidates(
            z_tilde,
            u,
            x_center,
            covariance_matrix,
            float(length),
            self.radial_mode,
            num_dim,
            self._cov_cache,
            gen=cov_gen,
        )
        return candidates

    def _generate_numpy(
        self,
        *,
        x_center: np.ndarray,
        num_dim: int,
        num_candidates: int,
        length: float,
        rng: Any,
        candidate_rv: CandidateRV,
        sobol_engine: Any | None,
        covariance_matrix: np.ndarray,
    ) -> np.ndarray:
        z = self._sample_raasp_whitened(
            num_candidates=num_candidates,
            num_dim=num_dim,
            length=length,
            rng=rng,
            candidate_rv=candidate_rv,
            sobol_engine=sobol_engine,
        )
        chol = np.linalg.cholesky(covariance_matrix)
        step = z @ chol.T
        candidates = _ray_scale_to_unit_box(x_center, x_center.reshape(1, -1) + step)
        delta = candidates - x_center.reshape(1, -1)
        dist2 = _mahalanobis_sq(delta, covariance_matrix)
        radius2 = float(length) ** 2
        bad = np.where(dist2 > radius2 * (1.0 + 1e-8))[0]
        if bad.size > 0:
            scale = np.sqrt(radius2 / np.maximum(dist2[bad], 1e-12))
            delta[bad] *= scale.reshape(-1, 1)
            candidates = _ray_scale_to_unit_box(x_center, x_center.reshape(1, -1) + delta)
        return candidates

    def generate(
        self,
        *,
        x_center: np.ndarray,
        num_dim: int,
        num_candidates: int,
        length: float,
        rng: Any,
        candidate_rv: CandidateRV | None,
        sobol_engine: Any | None,
        covariance_matrix: np.ndarray | None = None,
        cov_factor: np.ndarray | None = None,
        low_rank: _LowRankFactor | None = None,
        cov_gen: int = -1,
    ) -> np.ndarray:
        x_center = np.asarray(x_center, dtype=float).reshape(-1)
        rv = self.default_candidate_rv if candidate_rv is None else candidate_rv
        if low_rank is not None:
            return self._generate_low_rank(
                x_center=x_center,
                num_dim=num_dim,
                num_candidates=num_candidates,
                length=length,
                rng=rng,
                candidate_rv=rv,
                sobol_engine=sobol_engine,
                low_rank=low_rank,
            )
        if cov_factor is not None:
            return self._generate_cov_factor(
                x_center=x_center,
                num_dim=num_dim,
                num_candidates=num_candidates,
                length=length,
                rng=rng,
                candidate_rv=rv,
                sobol_engine=sobol_engine,
                cov_factor=cov_factor,
            )
        if self.use_accel and num_candidates * num_dim >= 250_000:
            candidates = self._generate_with_jax(
                x_center=x_center,
                num_dim=num_dim,
                num_candidates=num_candidates,
                length=length,
                rng=rng,
                candidate_rv=rv,
                sobol_engine=sobol_engine,
                covariance_matrix=covariance_matrix,
                cov_gen=cov_gen,
            )
            if candidates is not None:
                return candidates
        return self._generate_numpy(
            x_center=x_center,
            num_dim=num_dim,
            num_candidates=num_candidates,
            length=length,
            rng=rng,
            candidate_rv=rv,
            sobol_engine=sobol_engine,
            covariance_matrix=covariance_matrix,
        )


class _LengthPolicy:
    def reset(self) -> None:
        return

    def set_acceptance_ratio(self, *, pred: float, act: float, boundary_hit: bool) -> None:
        _ = pred, act, boundary_hit
        return

    @property
    def pending_rho(self) -> float | None:
        return None

    def apply_after_super_update(
        self,
        *,
        current_length: float,
        base_length: float,
        fixed_length: float | None,
        length_max: float,
    ) -> float:
        _ = base_length, fixed_length, length_max
        return float(current_length)


@dataclass
class _OptionCLengthPolicy(_LengthPolicy):
    """Classical trust-region acceptance-ratio controller.

    Uses rho = actual / predicted with thresholded shrink/expand multipliers,
    following standard trust-region ratio logic.
    """

    rho_bad: float
    rho_good: float
    gamma_down: float
    gamma_up: float
    _pending_rho: float | None = field(default=None, init=False)
    _pending_boundary_hit: bool = field(default=False, init=False)

    def reset(self) -> None:
        self._pending_rho = None
        self._pending_boundary_hit = False

    def set_acceptance_ratio(self, *, pred: float, act: float, boundary_hit: bool) -> None:
        eps = 1e-12
        denom = pred
        if not np.isfinite(denom) or abs(float(denom)) < eps:
            denom = eps if float(pred) >= 0.0 else -eps
        self._pending_rho = float(act) / float(denom)
        self._pending_boundary_hit = bool(boundary_hit)

    @property
    def pending_rho(self) -> float | None:
        return self._pending_rho

    def apply_after_super_update(
        self,
        *,
        current_length: float,
        base_length: float,
        fixed_length: float | None,
        length_max: float,
    ) -> float:
        rho = self._pending_rho
        length = float(current_length if fixed_length is not None else base_length)
        if rho is None:
            return float(length)
        if rho < float(self.rho_bad):
            length *= float(self.gamma_down)
        elif rho > float(self.rho_good) and self._pending_boundary_hit:
            length *= float(self.gamma_up)
        self._pending_rho = None
        self._pending_boundary_hit = False
        return float(min(length, length_max))
