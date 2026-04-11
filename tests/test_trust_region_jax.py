"""Tests for JAX-accelerated trust-region kernels.

Verifies that every JAX path produces results numerically close to the
NumPy reference implementation, and that the integration flags wire through
the full config → designer stack.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

import optimizer.trust_region_accel as accel
import optimizer.trust_region_utils as tru
from optimizer.trust_region_math import (
    _clip_to_unit_box,
    _ray_scale_to_unit_box,
)

pytestmark = pytest.mark.skipif(not accel.is_available(), reason="JAX not installed")


RNG = np.random.default_rng(42)
D = 32
N = 50


def _random_spd(dim: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((dim, dim))
    return a @ a.T + np.eye(dim) * 0.1


def _f32(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def _matmul_tol() -> dict[str, float]:
    accel_kind = accel.accel_name()
    if accel_kind in {"triton", "jax"}:
        return {"rtol": 3e-3, "atol": 1.5e-2}
    return {"rtol": 1e-3, "atol": 1e-5}


@pytest.fixture
def _force_jax_backend():
    old_backend = getattr(accel, "_ACCEL", None)
    accel.set_accel("jax")
    yield
    accel._ACCEL = old_backend


# ------------------------------------------------------------------
# Kernel-level tests
# ------------------------------------------------------------------


class TestMahalanobisSq:
    def test_matches_numpy_via_solve(self):
        cov = _random_spd(D, RNG)
        delta = RNG.standard_normal((N, D))
        delta32 = _f32(delta)
        cov32 = _f32(cov)
        expected = np.sum(delta32 * np.linalg.solve(cov32, delta32.T).T, axis=1, dtype=np.float32).astype(np.float64)
        got = accel.mahalanobis_sq_from_cov(delta, cov)
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-5)

    def test_matches_numpy_via_inv(self):
        cov = _random_spd(D, RNG)
        cov32 = _f32(cov)
        cov_inv = np.linalg.inv(cov32).astype(np.float64)
        delta = RNG.standard_normal((N, D))
        delta32 = _f32(delta)
        expected = np.sum((delta32 @ cov_inv.astype(np.float32)) * delta32, axis=1, dtype=np.float32).astype(np.float64)
        got = accel.mahalanobis_sq(delta, cov_inv)
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-5)

    def test_matches_numpy_via_factor(self):
        cov = _random_spd(D, RNG)
        chol32 = np.linalg.cholesky(_f32(cov))
        delta = RNG.standard_normal((N, D))
        delta32 = _f32(delta)
        expected = np.sum(np.linalg.solve(chol32, delta32.T).T ** 2, axis=1, dtype=np.float32).astype(np.float64)
        got = accel.mahalanobis_sq_from_factor(delta, chol32.astype(np.float64))
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-5)


class TestLowRankStep:
    def test_matches_numpy_matmul(self):
        rank = 8
        basis = RNG.standard_normal((D, rank))
        coeff = RNG.standard_normal((N, rank))
        expected = (_f32(coeff) @ _f32(basis).T).astype(np.float64)
        got = accel.low_rank_step(coeff, basis)
        np.testing.assert_allclose(got, expected, **_matmul_tol())

    def test_with_sparse_axis_matches_numpy(self):
        rank = 8
        basis = RNG.standard_normal((D, rank))
        coeff = RNG.standard_normal((N, rank))
        z = RNG.uniform(-0.5, 0.5, size=(N, D))
        scale = 0.125
        expected = ((_f32(coeff) @ _f32(basis).T) + np.float32(scale) * _f32(z)).astype(np.float64)
        got = accel.low_rank_step_with_sparse(coeff, basis, z, scale)
        np.testing.assert_allclose(got, expected, **_matmul_tol())

    def test_fused_low_rank_candidates_matches_reference(self):
        rank = 8
        basis = RNG.standard_normal((D, rank))
        coeff = RNG.standard_normal((N, rank))
        z = RNG.uniform(-0.5, 0.5, size=(N, D))
        scale = 0.125
        x_center = RNG.uniform(0.1, 0.9, size=(D,))
        length = 0.8
        step = ((_f32(coeff) @ _f32(basis).T) + np.float32(scale) * _f32(z)) * np.float32(length)
        expected = _clip_to_unit_box(x_center, step.astype(np.float64))
        got = accel.fused_low_rank_candidates(coeff, basis, z, scale, x_center, length)
        np.testing.assert_allclose(got, expected, **_matmul_tol())


class TestLowRankMetric:
    def test_matches_numpy_woodbury(self):
        rank = 8
        basis = RNG.standard_normal((D, rank))
        beta = RNG.uniform(0.1, 2.0, size=(rank,))
        inv_alpha = float(RNG.uniform(0.5, 5.0))
        delta = RNG.standard_normal((N, D))
        delta32 = _f32(delta)
        basis32 = _f32(basis)
        beta32 = _f32(beta)
        proj = delta32 @ basis32
        expected = (inv_alpha * np.sum(delta32 * delta32, axis=1) - np.sum(proj * proj * beta32, axis=1)).astype(np.float64)
        got = accel.low_rank_metric(delta, basis, beta, inv_alpha)
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-5)


class TestClipToUnitBox:
    def test_matches_numpy_reference(self):
        x_center = RNG.uniform(0.1, 0.9, size=(D,))
        step = RNG.standard_normal((N, D)) * 0.3
        expected = _clip_to_unit_box(x_center, step)
        got = accel.clip_to_unit_box(x_center, step)
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)

    def test_all_inside_unit_box(self):
        x_center = RNG.uniform(0.1, 0.9, size=(D,))
        step = RNG.standard_normal((N, D)) * 0.5
        result = accel.clip_to_unit_box(x_center, step)
        assert np.all(result >= -1e-7)
        assert np.all(result <= 1.0 + 1e-7)


class TestRayScaleToUnitBox:
    def test_matches_numpy_reference(self):
        x_center = RNG.uniform(0.1, 0.9, size=(D,))
        x = x_center + RNG.standard_normal((N, D)) * 0.3
        expected = _ray_scale_to_unit_box(x_center, x)
        got = accel.ray_scale_to_unit_box(x_center, x)
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


class TestCholesky:
    def test_matches_numpy(self):
        cov = _random_spd(D, RNG)
        expected = np.linalg.cholesky(_f32(cov)).astype(np.float64)
        got = accel.cholesky(cov)
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-5)


class TestMatmul:
    def test_matches_numpy(self):
        a = RNG.standard_normal((N, D))
        b = RNG.standard_normal((D, D))
        expected = (_f32(a) @ _f32(b)).astype(np.float64)
        got = accel.matmul(a, b)
        np.testing.assert_allclose(got, expected, **_matmul_tol())


class TestFusedEllipsoid:
    def test_fused_whitened_ellipsoid_matches_composed_path(self):
        cov = _random_spd(D, RNG)
        mask = RNG.random((N, D)) < 0.3
        base = RNG.uniform(-1.0, 1.0, size=(N, D))
        z_tilde = (_f32(base) * _f32(mask)).astype(np.float64)
        u = RNG.random(N)
        x_center = RNG.uniform(0.1, 0.9, size=(D,))
        length = 0.8
        cache = accel.CovCache()
        z = accel.whitened_sample(z_tilde, u, length, "ball_uniform", D)
        expected = accel.fused_ellipsoid_generate(z, x_center, cov, length, cache, gen=0)
        cache.invalidate()
        got = accel.fused_whitened_ellipsoid_candidates(
            z_tilde,
            u,
            x_center,
            cov,
            length,
            "ball_uniform",
            D,
            cache,
            gen=1,
        )
        assert got is not None
        np.testing.assert_allclose(got, expected, **_matmul_tol())

    def test_fused_sobol_ellipsoid_matches_whitened_path(self):
        if not hasattr(accel.current_accel(), "fused_sobol_ellipsoid_candidates"):
            pytest.skip("accel has no fused Sobol ellipsoid path")
        normal_pairs = (D + 1) // 2
        total_cols = D + 2 * normal_pairs + 1
        samples = RNG.random((N, total_cols), dtype=np.float32)
        prob = 1.0
        z_tilde, u = tru._whitened_inputs_from_sobol_samples(samples, num_dim=D, prob=prob, rng=RNG)
        cov = _random_spd(D, RNG)
        x_center = RNG.uniform(0.1, 0.9, size=(D,))
        length = 0.8
        cache = accel.CovCache()
        expected = accel.fused_whitened_ellipsoid_candidates(
            z_tilde,
            u,
            x_center,
            cov,
            length,
            "ball_uniform",
            D,
            cache,
            gen=0,
        )
        cache.invalidate()
        got = accel.fused_sobol_ellipsoid_candidates(
            samples,
            x_center,
            cov,
            length,
            "ball_uniform",
            D,
            prob,
            cache,
            gen=1,
        )
        assert got is not None
        np.testing.assert_allclose(got, expected, **_matmul_tol())


class TestForcedJAXSmoke:
    def test_forced_jax_backend_executes_matmul(self, _force_jax_backend):
        a = RNG.standard_normal((N, D))
        b = RNG.standard_normal((D, D))
        assert accel.accel_name() == "jax"
        got = accel.matmul(a, b)
        expected = (_f32(a) @ _f32(b)).astype(np.float64)
        np.testing.assert_allclose(got, expected, **_matmul_tol())


class TestAccelEnvOverride:
    def test_env_can_force_accel_selection(self, monkeypatch):
        old_backend = getattr(accel, "_ACCEL", None)
        monkeypatch.setenv("YUBO_TR_ACCEL", "jax")
        monkeypatch.setitem(accel.ACCEL_MODULES, "mlx", types.SimpleNamespace(available=lambda: True))
        monkeypatch.setitem(accel.ACCEL_MODULES, "triton", types.SimpleNamespace(available=lambda: True))
        monkeypatch.setitem(accel.ACCEL_MODULES, "jax", types.SimpleNamespace(available=lambda: True))
        accel._ACCEL = None
        assert accel.accel_name() == "jax"
        accel._ACCEL = old_backend

    def test_accel_override_restores_previous_accel(self, monkeypatch):
        old_backend = getattr(accel, "_ACCEL", None)
        monkeypatch.setitem(accel.ACCEL_MODULES, "jax", types.SimpleNamespace(available=lambda: True))
        monkeypatch.setitem(accel.ACCEL_MODULES, "mlx", types.SimpleNamespace(available=lambda: False))
        monkeypatch.setitem(accel.ACCEL_MODULES, "triton", types.SimpleNamespace(available=lambda: False))
        accel._ACCEL = old_backend
        with accel.accel_override("jax"):
            assert accel.accel_name() == "jax"
        assert getattr(accel, "_ACCEL", None) == old_backend


# ------------------------------------------------------------------
# Integration: config wiring
# ------------------------------------------------------------------


class TestConfigWiring:
    def test_metric_shaped_config_accepts_use_accel(self):
        from optimizer.trust_region_config import MetricShapedTRConfig

        cfg = MetricShapedTRConfig(geometry="enn_metr", use_accel=True)
        assert cfg.use_accel is True

    def test_metric_shaped_config_default_false(self):
        from optimizer.trust_region_config import MetricShapedTRConfig

        cfg = MetricShapedTRConfig()
        assert cfg.use_accel is False

    def test_build_metric_shaped_with_jax(self):
        from optimizer.trust_region_config import MetricShapedTRConfig

        cfg = MetricShapedTRConfig(geometry="enn_metr", use_accel=True)
        tr = cfg.build(num_dim=D, rng=RNG)
        assert tr.use_accel is True

    def test_build_true_ellipsoid_with_jax(self):
        from optimizer.trust_region_config import MetricShapedTRConfig

        cfg = MetricShapedTRConfig(geometry="enn_ellip", use_accel=True)
        tr = cfg.build(num_dim=D, rng=RNG)
        assert tr.use_accel is True

    def test_build_box_ignores_jax(self):
        from optimizer.trust_region_config import MetricShapedTRConfig

        cfg = MetricShapedTRConfig(geometry="box", use_accel=True)
        tr = cfg.build(num_dim=D, rng=RNG)
        assert not hasattr(tr, "use_accel") or not tr.use_accel


# ------------------------------------------------------------------
# Integration: candidate generation with JAX enabled
# ------------------------------------------------------------------


class TestCandidateGenerationJAX:
    def test_metric_shaped_generates_valid_candidates(self):
        from enn.turbo.config.candidate_rv import CandidateRV

        from optimizer.trust_region_config import MetricShapedTRConfig

        cfg = MetricShapedTRConfig(geometry="enn_metr", use_accel=True)
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

        cfg = MetricShapedTRConfig(geometry="enn_ellip", use_accel=True)
        tr = cfg.build(num_dim=D, rng=RNG, candidate_rv=CandidateRV.UNIFORM)
        dx = RNG.standard_normal((20, D))
        w = np.ones(20) / 20.0
        tr.set_geometry(dx, w)
        x_center = RNG.uniform(0.1, 0.9, size=(D,))
        candidates = tr.generate_candidates(x_center, None, 30, rng=RNG)
        assert candidates.shape == (30, D)
        assert np.all(candidates >= -1e-7)
        assert np.all(candidates <= 1.0 + 1e-7)
