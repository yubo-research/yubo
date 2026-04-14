"""
Compatibility facade for trust-region geometry, sampling, and length policies.

This module intentionally preserves a small legacy import surface that tests and
older callers still reach into directly. New implementation code should prefer
the dedicated modules:
- `trust_region_geometry.py`
- `trust_region_step_samplers.py`
- `trust_region_length_policies.py`

References:
- TuRBO: Eriksson et al., NeurIPS 2019 (Scalable Global Optimization via Local Bayesian Optimization),
  https://arxiv.org/abs/1910.01739
- Classical trust-region ratio update: Conn, Gould, Toint, *Trust Region Methods* (SIAM, 2000),
  https://doi.org/10.1137/1.9780898719857
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from enn.turbo.config.candidate_rv import CandidateRV
from enn.turbo.turbo_utils import generate_raasp_candidates

import optimizer.trust_region_accel as _accel
import optimizer.trust_region_sampling_utils as _sampling
from optimizer.trust_region_math import _ray_scale_to_unit_box


def _generate_block_raasp_candidates(*args, **kwargs):
    return _sampling._generate_block_raasp_candidates(*args, **kwargs)


def _whitened_inputs_from_sobol_samples(*args, **kwargs):
    return _sampling._whitened_inputs_from_sobol_samples(*args, **kwargs)


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
        rng,
        candidate_rv: CandidateRV | None,
        sobol_engine,
        num_pert: int,
        build_step,
        build_candidates=None,
        cov_factor: np.ndarray | None = None,
    ) -> np.ndarray:
        z_center = np.zeros(num_dim, dtype=float)
        lb = -0.5 * np.ones(num_dim, dtype=float)
        ub = 0.5 * np.ones(num_dim, dtype=float)
        rv = self.default_candidate_rv if candidate_rv is None else candidate_rv
        rv_name = _sampling._candidate_rv_name(rv)
        if self.use_accel and rv_name in {"uniform", "gpu_uniform"}:
            z = _sampling._generate_raasp_candidates_fast_uniform(
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
            z = _sampling._generate_raasp_candidates_fast_sobol(
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
        if cov_factor is not None and self.use_accel:
            return _accel.fused_metric_candidates(
                z,
                np.asarray(x_center, dtype=float),
                cov_factor,
                float(length),
            )
        if build_candidates is not None:
            return build_candidates(
                z,
                rng,
                np.asarray(x_center, dtype=float),
                float(length),
            )
        step = build_step(z, rng) * float(length)
        x_center_arr = np.asarray(x_center, dtype=float)
        if self.use_accel:
            return _accel.clip_to_unit_box(x_center_arr, step)
        return _ray_scale_to_unit_box(x_center_arr, x_center_arr.reshape(1, -1) + step)


__all__ = [
    "generate_raasp_candidates",
    "_generate_block_raasp_candidates",
    "_whitened_inputs_from_sobol_samples",
    "_AxisAlignedStepSampler",
]
