"""
Candidate-generation helpers for shaped trust regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
from enn.turbo.config.candidate_rv import CandidateRV

import optimizer.trust_region_accel as _accel
from optimizer.trust_region_math import _LowRankFactor, _mahalanobis_sq, _mahalanobis_sq_from_factor, _ray_scale_to_unit_box
from optimizer.trust_region_sampling_utils import (
    _candidate_rv_name,
    _draw_sobol_prefix,
    _low_rank_mahalanobis_sq,
    _low_rank_symmetric_sqrt_step,
    _sample_whitened_inputs,
    _whitened_sample_numpy,
)

RadialMode = Literal["ball_uniform", "boundary"]


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
        prob = float(self.p_raasp)
        z_tilde, u = _sample_whitened_inputs(
            num_candidates=num_candidates,
            num_dim=num_dim,
            rng=rng,
            candidate_rv=candidate_rv,
            prob=prob,
            sobol_engine=sobol_engine,
        )
        if self.use_accel:
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
        if self.use_accel:
            dist2 = _accel.mahalanobis_sq_from_factor(delta, cov_factor)
        else:
            dist2 = _mahalanobis_sq_from_factor(delta, np.asarray(cov_factor, dtype=float))
        radius2 = float(length) ** 2
        bad = np.where(dist2 > radius2 * (1.0 + 1e-8))[0]
        if bad.size > 0:
            scale = np.sqrt(radius2 / np.maximum(dist2[bad], 1e-12))
            delta[bad] *= scale.reshape(-1, 1)
            candidates = _ray_scale_to_unit_box(x_center, x_center.reshape(1, -1) + delta)
        return candidates

    def _generate_with_accel(
        self,
        *,
        x_center: np.ndarray,
        num_dim: int,
        num_candidates: int,
        length: float,
        rng: Any,
        candidate_rv: CandidateRV,
        sobol_engine: Any | None,
        covariance_matrix: np.ndarray | Callable[[], np.ndarray],
        cov_gen: int,
    ) -> np.ndarray | None:
        if not self.use_accel:
            return None
        rv_name = _candidate_rv_name(candidate_rv)
        cov = covariance_matrix() if callable(covariance_matrix) else covariance_matrix
        if rv_name == "sobol":
            if sobol_engine is None:
                raise ValueError("sobol_engine required for CandidateRV.SOBOL")
            sobol_samples = np.asarray(_draw_sobol_prefix(sobol_engine, int(num_candidates)), dtype=np.float32)
            candidates = _accel.fused_sobol_ellipsoid_candidates(
                sobol_samples,
                x_center,
                cov,
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
        return _accel.fused_whitened_ellipsoid_candidates(
            z_tilde,
            u,
            x_center,
            cov,
            float(length),
            self.radial_mode,
            num_dim,
            self._cov_cache,
            gen=cov_gen,
        )

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
        covariance_matrix: np.ndarray | Callable[[], np.ndarray],
    ) -> np.ndarray:
        z = self._sample_raasp_whitened(
            num_candidates=num_candidates,
            num_dim=num_dim,
            length=length,
            rng=rng,
            candidate_rv=candidate_rv,
            sobol_engine=sobol_engine,
        )
        cov = covariance_matrix() if callable(covariance_matrix) else covariance_matrix
        chol = np.linalg.cholesky(cov)
        step = z @ chol.T
        candidates = _ray_scale_to_unit_box(x_center, x_center.reshape(1, -1) + step)
        delta = candidates - x_center.reshape(1, -1)
        dist2 = _mahalanobis_sq(delta, cov)
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
        covariance_matrix: np.ndarray | Callable[[], np.ndarray] | None = None,
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
        if self.use_accel:
            candidates = self._generate_with_accel(
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
