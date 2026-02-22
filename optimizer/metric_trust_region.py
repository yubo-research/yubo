from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from enn.turbo.config.candidate_rv import CandidateRV
from enn.turbo.turbo_trust_region import TurboTrustRegion

from optimizer.pc_rotation import compute_labcat_weighted_pca
from optimizer.trust_region_utils import (
    _AxisAlignedStepSampler,
    _LengthPolicy,
    _mahalanobis_sq,
    _MetricGeometryModel,
)

_GRADIENT_GEOMETRIES = {"enn_grad_metric_shaped", "enn_grad_true_ellipsoid"}


def _apply_fixed_length_to_tr(tr: TurboTrustRegion) -> None:
    fixed_length = getattr(tr.config, "fixed_length", None)
    if fixed_length is not None:
        tr.length = float(fixed_length)


@dataclass
class MetricShapedTrustRegion(TurboTrustRegion):
    candidate_rv: CandidateRV = CandidateRV.SOBOL
    metric_sampler: str = "full"
    metric_rank: int | None = None
    uses_custom_candidate_gen: bool = field(default=True, init=False)
    _geometry_model: _MetricGeometryModel = field(init=False, repr=False)
    _step_sampler: _AxisAlignedStepSampler = field(init=False, repr=False)
    _length_policy: _LengthPolicy = field(init=False, repr=False)
    has_geometry: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._geometry_model = _MetricGeometryModel(
            num_dim=self.num_dim,
            metric_sampler=self.metric_sampler,
            metric_rank=self.metric_rank,
            pc_rotation_mode=getattr(self.config, "pc_rotation_mode", None),
            pc_rank=getattr(self.config, "pc_rank", None),
        )
        self._step_sampler = _AxisAlignedStepSampler(default_candidate_rv=self.candidate_rv)
        self._length_policy = _LengthPolicy()
        self._sync_geometry_flags()
        _apply_fixed_length_to_tr(self)

    def _sync_geometry_flags(self) -> None:
        self.has_geometry = bool(self._geometry_model.has_geometry)

    def _fixed_length(self) -> float | None:
        return getattr(self.config, "fixed_length", None)

    def restart(self, rng: Any | None = None) -> None:
        super().restart(rng=rng)
        self._geometry_model.reset()
        self._length_policy.reset()
        self._sync_geometry_flags()
        _apply_fixed_length_to_tr(self)

    def update(self, y_obs: np.ndarray | Any, y_incumbent: np.ndarray | Any) -> None:
        base_length = float(self.length)
        super().update(y_obs, y_incumbent)
        adjusted_length = self._length_policy.apply_after_super_update(
            current_length=float(self.length),
            base_length=base_length,
            fixed_length=self._fixed_length(),
            length_max=float(self.config.length_max),
        )
        if self._fixed_length() is None:
            self.length = float(adjusted_length)
        _apply_fixed_length_to_tr(self)

    def needs_restart(self) -> bool:
        if self._fixed_length() is not None:
            return False
        return super().needs_restart()

    def set_geometry(self, delta_x: np.ndarray | Any, weights: np.ndarray | Any) -> None:
        self._geometry_model.set_geometry(delta_x=delta_x, weights=weights)
        self._sync_geometry_flags()

    def observe_pc_rotation_geometry(
        self,
        *,
        x_center: np.ndarray,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        maximize: bool = True,
    ) -> None:
        """Update geometry from LABCAT-style weighted PCA (see pc_rotation module).

        Uses all observations to compute principal-component rotation. Only applies
        when pc_rotation_mode is set in config.
        """
        if getattr(self.config, "pc_rotation_mode", None) is None:
            return
        mode = str(self.config.pc_rotation_mode)
        rank = getattr(self.config, "pc_rank", None)
        result = compute_labcat_weighted_pca(
            x_center=np.asarray(x_center, dtype=float).reshape(-1),
            x_obs=np.asarray(x_obs, dtype=float),
            y_obs=np.asarray(y_obs, dtype=float).reshape(-1),
            maximize=maximize,
            mode=mode,
            rank=rank,
        )
        self._geometry_model.update_from_pc_rotation(result)
        self._sync_geometry_flags()

    def needs_gradient_signal(self) -> bool:
        return False

    def observe_local_geometry(
        self,
        *,
        delta_x: np.ndarray | Any,
        weights: np.ndarray | Any,
        delta_y: np.ndarray | Any | None = None,
    ) -> None:
        _ = delta_y
        self.set_geometry(delta_x=delta_x, weights=weights)

    def generate_candidates(
        self,
        x_center: np.ndarray,
        lengthscales: np.ndarray | None,
        num_candidates: int,
        *,
        rng: Any,
        candidate_rv: CandidateRV | None = None,
        sobol_engine: Any | None = None,
        raasp_driver: Any | None = None,
        num_pert: int = 20,
    ) -> np.ndarray:
        _ = raasp_driver
        if lengthscales is not None:
            raise ValueError("lengthscales are not supported for metric-shaped trust regions")
        num_candidates = int(num_candidates)
        if num_candidates <= 0:
            raise ValueError(num_candidates)
        num_dim = int(self.num_dim)
        x_center = np.asarray(x_center, dtype=float).reshape(-1)
        if x_center.shape != (num_dim,):
            raise ValueError((x_center.shape, num_dim))
        candidates = self._step_sampler.generate(
            x_center=x_center,
            length=float(self.length),
            num_dim=num_dim,
            num_candidates=num_candidates,
            rng=rng,
            candidate_rv=candidate_rv,
            sobol_engine=sobol_engine,
            num_pert=num_pert,
            build_step=self._geometry_model.build_step,
        )
        if candidates.shape != (num_candidates, num_dim):
            raise RuntimeError((candidates.shape, (num_candidates, num_dim)))
        return candidates

    def _covariance_matrix(self) -> np.ndarray:
        return self._geometry_model.covariance_matrix(
            jitter=float(getattr(self.config, "shape_jitter", 1e-6)),
        )

    def _mahalanobis_sq(self, delta: np.ndarray, cov: np.ndarray) -> np.ndarray:
        return _mahalanobis_sq(delta, cov)

    def set_acceptance_ratio(self, *, pred: float, act: float, boundary_hit: bool) -> None:
        self._length_policy.set_acceptance_ratio(pred=pred, act=act, boundary_hit=boundary_hit)

    @property
    def _pending_rho(self) -> float | None:
        return self._length_policy.pending_rho


@dataclass
class ENNMetricShapedTrustRegion(MetricShapedTrustRegion):
    has_enn_geometry: bool = field(default=False, init=False)

    def _sync_geometry_flags(self) -> None:
        super()._sync_geometry_flags()
        self.has_enn_geometry = self.has_geometry

    def set_geometry(self, delta_x: np.ndarray | Any, weights: np.ndarray | Any) -> None:
        super().set_geometry(delta_x=delta_x, weights=weights)

    def set_gradient_geometry(
        self,
        delta_x: np.ndarray | Any,
        delta_y: np.ndarray | Any,
        weights: np.ndarray | Any,
        *,
        eps_norm: float = 1e-12,
    ) -> None:
        self._geometry_model.set_gradient_geometry(
            delta_x=delta_x,
            delta_y=delta_y,
            weights=weights,
            eps_norm=eps_norm,
        )
        self._sync_geometry_flags()

    def set_analytic_gradient_geometry(self, grad: np.ndarray | Any) -> None:
        """Set geometry from analytic gradient ∇μ(x). Elongates trust region along gradient."""
        g = np.asarray(grad, dtype=float).reshape(-1)
        if g.shape[0] != self.num_dim or not np.any(np.isfinite(g)):
            return
        g_norm = float(np.linalg.norm(g))
        if g_norm <= 0.0:
            return
        self.set_gradient_geometry(
            delta_x=g.reshape(1, -1),
            delta_y=np.array([g_norm]),
            weights=np.array([1.0]),
        )

    def needs_gradient_signal(self) -> bool:
        geometry = str(getattr(self.config, "geometry", ""))
        return geometry in _GRADIENT_GEOMETRIES

    def observe_local_geometry(
        self,
        *,
        delta_x: np.ndarray | Any | None = None,
        weights: np.ndarray | Any | None = None,
        delta_y: np.ndarray | Any | None = None,
        grad: np.ndarray | Any | None = None,
    ) -> None:
        if self.needs_gradient_signal():
            if grad is not None:
                self.set_analytic_gradient_geometry(grad)
                return
            if delta_y is None:
                return
            self.set_gradient_geometry(
                delta_x=delta_x,
                delta_y=delta_y,
                weights=weights,
            )
            return
        if delta_x is not None and weights is not None:
            self.set_geometry(delta_x=delta_x, weights=weights)
