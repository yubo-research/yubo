from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from enn.turbo.config.candidate_rv import CandidateRV

import optimizer.trust_region_accel as _accel
from optimizer.metric_trust_region import ENNMetricShapedTrustRegion, MetricShapedTrustRegion, _apply_fixed_length_to_tr
from optimizer.trust_region_math import _mahalanobis_sq_from_factor, _ray_scale_to_unit_box
from optimizer.trust_region_sampling_utils import _block_indices_from_group, _low_rank_mahalanobis_sq, _sample_block_groups
from optimizer.trust_region_utils import (
    _LengthPolicy,
    _OptionCLengthPolicy,
    _TrueEllipsoidGeometryModel,
    _TrueEllipsoidStepSampler,
)


@dataclass
class ENNTrueEllipsoidalTrustRegion(ENNMetricShapedTrustRegion):
    _prev_incumbent_value: float | None = field(default=None, init=False, repr=False)
    _prev_incumbent_x: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._geometry_model = _TrueEllipsoidGeometryModel(
            num_dim=self.num_dim,
            covmat=self.covmat,
            metric_rank=self.metric_rank,
            pc_rotation_mode=getattr(self.config, "pc_rotation_mode", None),
            pc_rank=getattr(self.config, "pc_rank", None),
            use_accel=self.use_accel,
            update_option=getattr(self.config, "update_option", "option_a"),
            shape_period=int(getattr(self.config, "shape_period", 5)),
            shape_ema=float(getattr(self.config, "shape_ema", 0.2)),
            shape_jitter=float(getattr(self.config, "shape_jitter", 1e-6)),
            shape_kappa_max=float(getattr(self.config, "shape_kappa_max", 1e4)),
        )
        self._step_sampler = _TrueEllipsoidStepSampler(
            default_candidate_rv=self.candidate_rv,
            p_raasp=float(getattr(self.config, "p_raasp", 0.2)),
            radial_mode=str(getattr(self.config, "radial_mode", "ball_uniform")),
            use_accel=self.use_accel,
        )
        if str(getattr(self.config, "update_option", "option_a")) == "option_c":
            self._length_policy = _OptionCLengthPolicy(
                rho_bad=float(getattr(self.config, "rho_bad", 0.25)),
                rho_good=float(getattr(self.config, "rho_good", 0.75)),
                gamma_down=float(getattr(self.config, "gamma_down", 0.5)),
                gamma_up=float(getattr(self.config, "gamma_up", 2.0)),
            )
        else:
            self._length_policy = _LengthPolicy()
        self._reset_true_ellipsoid_state()
        self._sync_geometry_flags()
        _apply_fixed_length_to_tr(self)

    def _reset_true_ellipsoid_state(self) -> None:
        self._prev_incumbent_value = None
        self._prev_incumbent_x = None

    def restart(self, rng: Any | None = None) -> None:
        super().restart(rng=rng)
        self._reset_true_ellipsoid_state()

    def record_incumbent_transition(
        self,
        *,
        x_center: np.ndarray,
        y_value: float,
    ) -> tuple[float, np.ndarray] | None:
        x_curr = np.asarray(x_center, dtype=float).reshape(-1).copy()
        prev_val = self._prev_incumbent_value
        prev_x = self._prev_incumbent_x
        self._prev_incumbent_value = float(y_value)
        self._prev_incumbent_x = x_curr
        if prev_val is None or prev_x is None:
            return None
        return float(prev_val), np.asarray(prev_x, dtype=float).reshape(-1)

    def observe_incumbent_transition(
        self,
        *,
        x_center: np.ndarray,
        y_value: float,
        predict_delta,
    ) -> None:
        curr_x = np.asarray(x_center, dtype=float).reshape(-1)
        prev_pair = self.record_incumbent_transition(x_center=curr_x, y_value=float(y_value))
        if prev_pair is None:
            return
        prev_val, prev_x = prev_pair
        act = float(y_value - float(prev_val))
        pred = predict_delta(np.asarray(prev_x, dtype=float).reshape(-1), curr_x)
        eps = 1e-12
        if pred is None or (not np.isfinite(float(pred))) or abs(float(pred)) < eps:
            return
        delta = curr_x - np.asarray(prev_x, dtype=float).reshape(-1)
        if self._geometry_model.covmat == "low_rank":
            if self.use_accel:
                dist2 = float(
                    _low_rank_mahalanobis_sq(
                        delta.reshape(1, -1),
                        self._geometry_model.low_rank,
                    )[0]
                )
            else:
                with _accel.accel_override(""):
                    dist2 = float(
                        _low_rank_mahalanobis_sq(
                            delta.reshape(1, -1),
                            self._geometry_model.low_rank,
                        )[0]
                    )
        elif self._geometry_model.covmat == "dense":
            mahal = getattr(self._geometry_model, "mahalanobis_sq")
            if getattr(mahal, "__func__", mahal) is not getattr(type(self._geometry_model), "mahalanobis_sq"):
                dist2 = float(
                    mahal(
                        delta.reshape(1, -1),
                        jitter=float(getattr(self.config, "shape_jitter", 1e-6)),
                    )[0]
                )
            else:
                dist2 = float(
                    _mahalanobis_sq_from_factor(
                        delta.reshape(1, -1),
                        np.asarray(self._geometry_model.cov_factor, dtype=float),
                    )[0]
                )
        else:
            dist2 = float(
                self._geometry_model.mahalanobis_sq(
                    delta.reshape(1, -1),
                    jitter=float(getattr(self.config, "shape_jitter", 1e-6)),
                )[0]
            )
        dist = float(np.sqrt(max(0.0, dist2)))
        length = float(getattr(self, "length", 1.0))
        tol = float(getattr(self.config, "boundary_tol", 0.1))
        boundary_hit = dist >= max(0.0, (1.0 - tol) * length)
        self.set_acceptance_ratio(pred=float(pred), act=act, boundary_hit=boundary_hit)

    def generate_candidates(
        self,
        x_center: np.ndarray,
        lengthscales: np.ndarray | None,
        num_candidates: int,
        *,
        rng: Any,
        candidate_rv: Any | None = None,
        sobol_engine: Any | None = None,
        raasp_driver: Any | None = None,
        num_pert: int = 20,
    ) -> np.ndarray:
        _ = raasp_driver, num_pert
        if lengthscales is not None:
            raise ValueError("lengthscales are not supported for metric-shaped trust regions")
        num_candidates = int(num_candidates)
        if num_candidates <= 0:
            raise ValueError(num_candidates)
        num_dim = int(self.num_dim)
        x_center = np.asarray(x_center, dtype=float).reshape(-1)
        if x_center.shape != (num_dim,):
            raise ValueError((x_center.shape, num_dim))
        low_rank = self._geometry_model.low_rank if self._geometry_model.covmat == "low_rank" else None
        cov = self._covariance_matrix if low_rank is None else None
        cov_factor = self._geometry_model.cov_factor if self._geometry_model.covmat == "dense" else None
        candidates = self._step_sampler.generate(
            x_center=x_center,
            num_dim=num_dim,
            num_candidates=num_candidates,
            length=float(self.length),
            rng=rng,
            candidate_rv=candidate_rv,
            sobol_engine=sobol_engine,
            covariance_matrix=cov,
            cov_factor=cov_factor,
            low_rank=low_rank,
            cov_gen=self._geometry_model._cov_gen,
        )
        if getattr(self, "module_block_slices", ()):
            return self._generate_module_candidates(
                x_center=x_center,
                num_candidates=num_candidates,
                rng=rng,
                candidate_rv=candidate_rv,
                sobol_engine=sobol_engine,
            )
        return candidates

    def _generate_module_candidates(
        self,
        *,
        x_center: np.ndarray,
        num_candidates: int,
        rng: Any,
        candidate_rv: CandidateRV | None,
        sobol_engine: Any | None,
    ) -> np.ndarray:
        rv = self.candidate_rv if candidate_rv is None else candidate_rv
        jitter = float(getattr(self.config, "shape_jitter", 1e-6))
        groups = _sample_block_groups(
            num_candidates=num_candidates,
            rng=rng,
            block_slices=tuple(getattr(self, "module_block_slices", ())),
            block_prob=float(getattr(self, "module_block_prob", 0.0)),
        )
        candidates = np.tile(np.asarray(x_center, dtype=float).reshape(1, -1), (int(num_candidates), 1))
        radius2 = float(self.length) ** 2
        for block_group, rows in groups.items():
            indices = _block_indices_from_group(tuple(getattr(self, "module_block_slices", ())), block_group)
            if indices.size <= 0:
                continue
            x_sub = np.asarray(x_center, dtype=float)[indices]
            factor_sub = self._geometry_model.restricted_factor(indices, jitter=jitter)
            z_sub = self._step_sampler._sample_raasp_whitened(
                num_candidates=int(rows.size),
                num_dim=int(indices.size),
                length=float(self.length),
                rng=rng,
                candidate_rv=rv,
                sobol_engine=sobol_engine,
            )
            cand_sub = _ray_scale_to_unit_box(
                x_sub,
                x_sub.reshape(1, -1) + z_sub @ factor_sub.T,
            )
            delta_sub = cand_sub - x_sub.reshape(1, -1)
            dist2 = _mahalanobis_sq_from_factor(delta_sub, factor_sub)
            bad = np.where(dist2 > radius2 * (1.0 + 1e-8))[0]
            if bad.size > 0:
                scale = np.sqrt(radius2 / np.maximum(dist2[bad], 1e-12))
                delta_sub = delta_sub.copy()
                delta_sub[bad] *= scale.reshape(-1, 1)
                cand_sub = _ray_scale_to_unit_box(x_sub, x_sub.reshape(1, -1) + delta_sub)
            candidates[np.ix_(rows, indices)] = cand_sub
        return candidates


@dataclass
class ENNIsotropicTrustRegion(MetricShapedTrustRegion):
    """Fixed identity-metric trust region.

    This keeps the TuRBO length controller but freezes geometry at the
    identity metric, so proposals stay isotropic and do not learn anisotropy.
    """

    def needs_local_geometry(self) -> bool:
        return False

    def set_geometry(self, delta_x: np.ndarray | Any, weights: np.ndarray | Any) -> None:
        _ = delta_x, weights

    def set_gradient_geometry(
        self,
        delta_x: np.ndarray | Any,
        delta_y: np.ndarray | Any,
        weights: np.ndarray | Any,
        *,
        eps_norm: float = 1e-12,
    ) -> None:
        _ = delta_x, delta_y, weights, eps_norm

    def set_analytic_gradient_geometry(self, grad: np.ndarray | Any) -> None:
        _ = grad

    def observe_pc_rotation_geometry(
        self,
        *,
        x_center: np.ndarray,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        maximize: bool = True,
    ) -> None:
        _ = x_center, x_obs, y_obs, maximize

    def observe_local_geometry(
        self,
        *,
        delta_x: np.ndarray | Any | None = None,
        weights: np.ndarray | Any | None = None,
        delta_y: np.ndarray | Any | None = None,
        grad: np.ndarray | Any | None = None,
    ) -> None:
        _ = delta_x, weights, delta_y, grad
