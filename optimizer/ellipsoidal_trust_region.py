from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from optimizer.metric_trust_region import (
    ENNMetricShapedTrustRegion,
    _apply_fixed_length_to_tr,
)
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
            metric_sampler=self.metric_sampler,
            metric_rank=self.metric_rank,
            pc_rotation_mode=getattr(self.config, "pc_rotation_mode", None),
            pc_rank=getattr(self.config, "pc_rank", None),
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
        cov = self._covariance_matrix()
        delta = curr_x - np.asarray(prev_x, dtype=float).reshape(-1)
        solved = np.linalg.solve(cov, delta.reshape(-1, 1)).reshape(-1)
        dist = float(np.sqrt(max(0.0, np.dot(delta, solved))))
        length = float(getattr(self, "length", 1.0))
        tol = float(getattr(self.config, "boundary_tol", 0.1))
        boundary_hit = dist >= max(0.0, (1.0 - tol) * length)
        self.set_acceptance_ratio(pred=float(pred), act=act, boundary_hit=boundary_hit)

    def set_geometry(self, delta_x: np.ndarray | Any, weights: np.ndarray | Any) -> None:
        super().set_geometry(delta_x=delta_x, weights=weights)

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
        _ = sobol_engine, raasp_driver, num_pert
        if lengthscales is not None:
            raise ValueError("lengthscales are not supported for metric-shaped trust regions")
        num_candidates = int(num_candidates)
        if num_candidates <= 0:
            raise ValueError(num_candidates)
        num_dim = int(self.num_dim)
        x_center = np.asarray(x_center, dtype=float).reshape(-1)
        if x_center.shape != (num_dim,):
            raise ValueError((x_center.shape, num_dim))
        cov = self._covariance_matrix()
        return self._step_sampler.generate(
            x_center=x_center,
            num_dim=num_dim,
            num_candidates=num_candidates,
            length=float(self.length),
            rng=rng,
            candidate_rv=candidate_rv,
            covariance_matrix=cov,
        )
