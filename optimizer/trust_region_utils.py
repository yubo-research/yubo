from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
from enn.turbo.config.candidate_rv import CandidateRV
from enn.turbo.turbo_utils import generate_raasp_candidates

from optimizer.pc_rotation import PCRotationMode, PCRotationResult
from optimizer.trust_region_math import (
    _add_sparse_axis,
    _apply_full_factor,
    _clip_to_unit_box,
    _ensure_spd,
    _full_factor,
    _low_rank_factor,
    _LowRankFactor,
    _mahalanobis_sq,
    _normalize_weights,
    _ray_scale_to_unit_box,
    _trace_normalize,
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
    cov_factor: np.ndarray = field(init=False)
    low_rank: _LowRankFactor = field(init=False)
    has_geometry: bool = field(default=False, init=False)

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
        dx = np.asarray(delta_x, dtype=float)
        if dx.ndim != 2 or dx.shape[0] == 0:
            return
        if dx.shape[1] != self.num_dim:
            raise ValueError(f"delta_x has incompatible shape {dx.shape} for num_dim={self.num_dim}")
        dy = np.asarray(delta_y, dtype=float).reshape(-1)
        w = np.asarray(weights, dtype=float).reshape(-1)
        if dy.shape[0] != dx.shape[0] or w.shape[0] != dx.shape[0]:
            raise ValueError((dy.shape, w.shape, dx.shape))
        w = _normalize_weights(w)
        if w is None:
            return
        norms = np.linalg.norm(dx, axis=1)
        scale = np.abs(dy) / np.maximum(norms, float(eps_norm))
        scale = np.where(np.isfinite(scale), scale, 0.0)
        if not np.any(scale > 0.0):
            return
        centered = dx * (np.sqrt(w) * scale).reshape(-1, 1)
        cov = centered.T @ centered
        cov = _trace_normalize(cov, self.num_dim)
        self.update_from_cov(centered=centered, weights=w, cov=cov)

    def update_from_cov(
        self,
        *,
        centered: np.ndarray,
        weights: np.ndarray,
        cov: np.ndarray,
    ) -> None:
        _ = centered, weights
        lam_min = 1e-4
        lam_max = 1e4
        eps = 1e-6
        if self.metric_sampler == "full":
            factor = _full_factor(cov, dim=self.num_dim, lam_min=lam_min, lam_max=lam_max, eps=eps)
            if factor is None:
                return
            self.cov_factor = factor
            self.has_geometry = True
            return
        if self.metric_sampler != "low_rank":
            raise ValueError(f"Unknown metric_sampler: {self.metric_sampler!r}")
        rank_cap = int(self.metric_rank) if self.metric_rank is not None else None
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
        """Update geometry from LABCAT-style PC rotation (see pc_rotation.LABCAT_CITATION).

        Uses the principal directions and singular values to build an ellipsoidal
        trust region aligned with the weighted PCs. Full mode uses all PCs;
        low_rank mode uses top-k PCs with isotropic residual for the null space.
        """
        if not rotation_result.has_rotation:
            return
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
            step = np.zeros((z.shape[0], self.num_dim), dtype=float)
        else:
            coeff = rng.uniform(-0.5, 0.5, size=(z.shape[0], rank)) * sqrt_vals.reshape(1, -1)
            step = coeff @ basis.T
        _add_sparse_axis(step, z, float(self.low_rank.sqrt_alpha))
        return step

    def covariance_matrix(self, *, jitter: float) -> np.ndarray:
        if self.metric_sampler == "full":
            cov = self.cov_factor @ self.cov_factor.T
            return _ensure_spd(cov, jitter=jitter)
        low_rank = self.low_rank
        alpha = float(low_rank.sqrt_alpha) ** 2
        basis = np.asarray(low_rank.basis, dtype=float)
        lam = np.asarray(low_rank.sqrt_vals, dtype=float) ** 2
        cov = alpha * np.eye(self.num_dim, dtype=float)
        if basis.size > 0 and lam.size > 0:
            cov = cov + (basis * lam.reshape(1, -1)) @ basis.T
        return _ensure_spd(cov, jitter=jitter)


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
        cov_hat = _trace_normalize(cov_hat, self.num_dim)
        cov_hat = _ensure_spd(cov_hat, jitter=float(self.shape_jitter))
        if self.update_option == "option_b":
            eigvals, eigvecs = np.linalg.eigh(cov_hat)
            eigvals = np.maximum(eigvals, 1e-10)
            eigvals = np.clip(eigvals, np.max(eigvals) / float(self.shape_kappa_max), np.max(eigvals))
            cov_hat = (eigvecs * eigvals.reshape(1, -1)) @ eigvecs.T
            cov_hat = _trace_normalize(cov_hat, self.num_dim)
        eta = float(self.shape_ema)
        if self.ema_cov is None:
            self.ema_cov = cov_hat
        else:
            self.ema_cov = _ensure_spd(
                (1.0 - eta) * self.ema_cov + eta * cov_hat,
                jitter=float(self.shape_jitter),
            )
        self.update_from_cov(centered=centered, weights=w, cov=self.ema_cov)


@dataclass(frozen=True)
class _AxisAlignedStepSampler:
    default_candidate_rv: CandidateRV

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
    ) -> np.ndarray:
        z_center = np.zeros(num_dim, dtype=float)
        lb = -0.5 * np.ones(num_dim, dtype=float)
        ub = 0.5 * np.ones(num_dim, dtype=float)
        rv = self.default_candidate_rv if candidate_rv is None else candidate_rv
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
        step = build_step(z, rng) * float(length)
        return _clip_to_unit_box(np.asarray(x_center, dtype=float), step)


@dataclass(frozen=True)
class _TrueEllipsoidStepSampler:
    default_candidate_rv: CandidateRV
    p_raasp: float
    radial_mode: RadialMode

    def _sample_raasp_whitened(
        self,
        *,
        num_candidates: int,
        num_dim: int,
        length: float,
        rng: Any,
        candidate_rv: CandidateRV,
    ) -> np.ndarray:
        prob = float(self.p_raasp)
        mask = rng.random((num_candidates, num_dim)) < prob
        empty = np.where(np.sum(mask, axis=1) == 0)[0]
        if empty.size > 0:
            cols = rng.integers(0, num_dim, size=empty.size)
            mask[empty, cols] = True
        rv_name = str(getattr(candidate_rv, "value", candidate_rv)).lower()
        if rv_name in {"uniform", "gpu_uniform"}:
            base = rng.uniform(-1.0, 1.0, size=(num_candidates, num_dim))
        else:
            base = rng.normal(size=(num_candidates, num_dim))
        z_tilde = base * mask
        norms = np.linalg.norm(z_tilde, axis=1)
        tiny = norms <= 1e-12
        if np.any(tiny):
            idx_rows = np.where(tiny)[0]
            idx_cols = rng.integers(0, num_dim, size=idx_rows.size)
            signs = np.where(rng.random(idx_rows.size) < 0.5, -1.0, 1.0)
            z_tilde[idx_rows] = 0.0
            z_tilde[idx_rows, idx_cols] = signs
            norms = np.linalg.norm(z_tilde, axis=1)
        v = z_tilde / norms.reshape(-1, 1)
        u = rng.random(num_candidates)
        if self.radial_mode == "boundary":
            rho = 0.8 + 0.2 * u
        else:
            rho = np.power(u, 1.0 / float(max(num_dim, 1)))
        return float(length) * rho.reshape(-1, 1) * v

    def generate(
        self,
        *,
        x_center: np.ndarray,
        num_dim: int,
        num_candidates: int,
        length: float,
        rng: Any,
        candidate_rv: CandidateRV | None,
        covariance_matrix: np.ndarray,
    ) -> np.ndarray:
        x_center = np.asarray(x_center, dtype=float).reshape(-1)
        rv = self.default_candidate_rv if candidate_rv is None else candidate_rv
        rv_name = str(getattr(rv, "value", rv)).lower()
        if rv_name not in {"sobol", "uniform", "gpu_uniform"}:
            rv = CandidateRV.SOBOL
        z = self._sample_raasp_whitened(
            num_candidates=num_candidates,
            num_dim=num_dim,
            length=length,
            rng=rng,
            candidate_rv=rv,
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
