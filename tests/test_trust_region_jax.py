"""Tests for JAX-accelerated trust-region kernels.

Verifies that every JAX path produces results numerically close to the
NumPy reference implementation, and that the integration flags wire through
the full config → designer stack.
"""

from __future__ import annotations

import numpy as np
import pytest

import optimizer.trust_region_jax as jax_tr
from optimizer.trust_region_math import (
    _clip_to_unit_box,
    _mahalanobis_sq,
    _ray_scale_to_unit_box,
)

pytestmark = pytest.mark.skipif(not jax_tr.is_available(), reason="JAX not installed")


RNG = np.random.default_rng(42)
D = 32
N = 50


def _random_spd(dim: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((dim, dim))
    return a @ a.T + np.eye(dim) * 0.1


# ------------------------------------------------------------------
# Kernel-level tests
# ------------------------------------------------------------------


class TestMahalanobisSq:
    def test_matches_numpy_via_solve(self):
        cov = _random_spd(D, RNG)
        delta = RNG.standard_normal((N, D))
        expected = _mahalanobis_sq(delta, cov)
        got = jax_tr.mahalanobis_sq_from_cov(delta, cov)
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-5)

    def test_matches_numpy_via_inv(self):
        cov = _random_spd(D, RNG)
        cov_inv = np.linalg.inv(cov)
        delta = RNG.standard_normal((N, D))
        expected = _mahalanobis_sq(delta, cov)
        got = jax_tr.mahalanobis_sq(delta, cov_inv)
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-5)


class TestLowRankStep:
    def test_matches_numpy_matmul(self):
        rank = 8
        basis = RNG.standard_normal((D, rank))
        coeff = RNG.standard_normal((N, rank))
        expected = coeff @ basis.T
        got = jax_tr.low_rank_step(coeff, basis)
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


class TestLowRankMetric:
    def test_matches_numpy_woodbury(self):
        rank = 8
        basis = RNG.standard_normal((D, rank))
        beta = RNG.uniform(0.1, 2.0, size=(rank,))
        inv_alpha = float(RNG.uniform(0.5, 5.0))
        delta = RNG.standard_normal((N, D))
        proj = delta @ basis
        expected = inv_alpha * np.sum(delta * delta, axis=1) - np.sum(proj * proj * beta, axis=1)
        got = jax_tr.low_rank_metric(delta, basis, beta, inv_alpha)
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-5)


class TestClipToUnitBox:
    def test_matches_numpy_reference(self):
        x_center = RNG.uniform(0.1, 0.9, size=(D,))
        step = RNG.standard_normal((N, D)) * 0.3
        expected = _clip_to_unit_box(x_center, step)
        got = jax_tr.clip_to_unit_box(x_center, step)
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)

    def test_all_inside_unit_box(self):
        x_center = RNG.uniform(0.1, 0.9, size=(D,))
        step = RNG.standard_normal((N, D)) * 0.5
        result = jax_tr.clip_to_unit_box(x_center, step)
        assert np.all(result >= -1e-7)
        assert np.all(result <= 1.0 + 1e-7)


class TestRayScaleToUnitBox:
    def test_matches_numpy_reference(self):
        x_center = RNG.uniform(0.1, 0.9, size=(D,))
        x = x_center + RNG.standard_normal((N, D)) * 0.3
        expected = _ray_scale_to_unit_box(x_center, x)
        got = jax_tr.ray_scale_to_unit_box(x_center, x)
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


class TestCholesky:
    def test_matches_numpy(self):
        cov = _random_spd(D, RNG)
        expected = np.linalg.cholesky(cov)
        got = jax_tr.cholesky(cov)
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-5)


class TestMatmul:
    def test_matches_numpy(self):
        a = RNG.standard_normal((N, D))
        b = RNG.standard_normal((D, D))
        expected = a @ b
        got = jax_tr.matmul(a, b)
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-5)


# ------------------------------------------------------------------
# Integration: config wiring
# ------------------------------------------------------------------


class TestConfigWiring:
    def test_metric_shaped_config_accepts_use_jax_tr(self):
        from optimizer.trust_region_config import MetricShapedTRConfig

        cfg = MetricShapedTRConfig(geometry="enn_metric_shaped", use_jax_tr=True)
        assert cfg.use_jax_tr is True

    def test_metric_shaped_config_default_false(self):
        from optimizer.trust_region_config import MetricShapedTRConfig

        cfg = MetricShapedTRConfig()
        assert cfg.use_jax_tr is False

    def test_build_metric_shaped_with_jax(self):
        from optimizer.trust_region_config import MetricShapedTRConfig

        cfg = MetricShapedTRConfig(geometry="enn_metric_shaped", use_jax_tr=True)
        tr = cfg.build(num_dim=D, rng=RNG)
        assert tr.use_jax is True

    def test_build_true_ellipsoid_with_jax(self):
        from optimizer.trust_region_config import MetricShapedTRConfig

        cfg = MetricShapedTRConfig(geometry="enn_true_ellipsoid", use_jax_tr=True)
        tr = cfg.build(num_dim=D, rng=RNG)
        assert tr.use_jax is True

    def test_build_box_ignores_jax(self):
        from optimizer.trust_region_config import MetricShapedTRConfig

        cfg = MetricShapedTRConfig(geometry="box", use_jax_tr=True)
        tr = cfg.build(num_dim=D, rng=RNG)
        assert not hasattr(tr, "use_jax") or not tr.use_jax


# ------------------------------------------------------------------
# Integration: candidate generation with JAX enabled
# ------------------------------------------------------------------


class TestCandidateGenerationJAX:
    def test_metric_shaped_generates_valid_candidates(self):
        from enn.turbo.config.candidate_rv import CandidateRV

        from optimizer.trust_region_config import MetricShapedTRConfig

        cfg = MetricShapedTRConfig(geometry="enn_metric_shaped", use_jax_tr=True)
        tr = cfg.build(num_dim=D, rng=RNG, candidate_rv=CandidateRV.UNIFORM)
        dx = RNG.standard_normal((20, D))
        w = np.ones(20) / 20.0
        tr.set_geometry(dx, w)
        x_center = RNG.uniform(0.1, 0.9, size=(D,))
        candidates = tr.generate_candidates(x_center, None, 30, rng=RNG)
        assert candidates.shape == (30, D)
        assert np.all(candidates >= -1e-7)
        assert np.all(candidates <= 1.0 + 1e-7)

    def test_true_ellipsoid_generates_valid_candidates(self):
        from enn.turbo.config.candidate_rv import CandidateRV

        from optimizer.trust_region_config import MetricShapedTRConfig

        cfg = MetricShapedTRConfig(geometry="enn_true_ellipsoid", use_jax_tr=True)
        tr = cfg.build(num_dim=D, rng=RNG, candidate_rv=CandidateRV.UNIFORM)
        dx = RNG.standard_normal((20, D))
        w = np.ones(20) / 20.0
        tr.set_geometry(dx, w)
        x_center = RNG.uniform(0.1, 0.9, size=(D,))
        candidates = tr.generate_candidates(x_center, None, 30, rng=RNG)
        assert candidates.shape == (30, D)
        assert np.all(candidates >= -1e-7)
        assert np.all(candidates <= 1.0 + 1e-7)
