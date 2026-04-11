import types

import numpy as np
import pytest
from enn.turbo.config.candidate_rv import CandidateRV

from optimizer import trust_region_utils as tru
from optimizer.pc_rotation import PCRotationResult


def test_metric_geometry_model_full_mode_paths():
    model = tru._MetricGeometryModel(num_dim=3, metric_sampler="full", metric_rank=None)
    model.reset()
    assert model.has_geometry is False

    with pytest.raises(ValueError, match="incompatible shape"):
        model.set_geometry(np.zeros((4, 2), dtype=float), np.ones((4,), dtype=float))

    dx = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=float)
    w = np.array([1.0, 2.0, 3.0], dtype=float)
    model.set_geometry(dx, w)
    assert model.has_geometry is True

    step = model.build_step(np.array([[0.2, -0.1, 0.0]], dtype=float), np.random.default_rng(0))
    assert step.shape == (1, 3)
    cov = model.covariance_matrix(jitter=1e-8)
    np.linalg.cholesky(cov)


def test_metric_geometry_model_low_rank_and_pc_rotation_paths():
    model = tru._MetricGeometryModel(
        num_dim=4,
        metric_sampler="low_rank",
        metric_rank=2,
        pc_rotation_mode="full",
        pc_rank=2,
    )
    dx = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    dy = np.array([1.0, -1.0, 2.0, -2.0], dtype=float)
    w = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    model.set_gradient_geometry(dx, dy, w)
    assert model.has_geometry is True

    with pytest.raises(ValueError):
        model.set_gradient_geometry(dx, np.array([1.0]), w)

    rr_bad = PCRotationResult(
        center=np.zeros((4,), dtype=float),
        basis=np.zeros((2, 2), dtype=float),
        singular_values=np.ones((2,), dtype=float),
        has_rotation=True,
    )
    model.update_from_pc_rotation(rr_bad)

    rr = PCRotationResult(
        center=np.zeros((4,), dtype=float),
        basis=np.eye(4, dtype=float),
        singular_values=np.array([2.0, 1.5, 1.0, 0.5], dtype=float),
        has_rotation=True,
    )
    model.update_from_pc_rotation(rr)
    assert model.has_geometry is True

    z = np.array([[1.0, 0.0, 1.0, 0.0]], dtype=float)
    step = model.build_step(z, np.random.default_rng(0))
    assert step.shape == (1, 4)
    cov = model.covariance_matrix(jitter=1e-8)
    np.linalg.cholesky(cov)

    model.metric_sampler = "bad"
    with pytest.raises(ValueError, match="Unknown metric_sampler"):
        model.update_from_cov(centered=dx, weights=np.ones((4,), dtype=float), cov=np.eye(4, dtype=float))


def test_low_rank_factor_from_cov_uses_dominant_eigenspace():
    cov = np.diag([9.0, 3.0, 1.0, 0.5]).astype(float)
    low_rank = tru._low_rank_factor_from_cov(
        cov,
        dim=4,
        lam_min=1e-4,
        lam_max=1e4,
        eps=1e-6,
        rank_cap=1,
    )
    assert low_rank is not None
    assert low_rank.basis.shape == (4, 1)
    assert np.allclose(np.abs(low_rank.basis[:, 0]), np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
    assert low_rank.sqrt_alpha > 0.0


def test_gradient_geometry_low_rank_uses_single_weighting_pass():
    model = tru._MetricGeometryModel(
        num_dim=4,
        metric_sampler="low_rank",
        metric_rank=2,
    )
    dx = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ],
        dtype=float,
    )
    dy = np.array([1.0, 0.5, 0.25, 0.125], dtype=float)
    weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    prepared = tru._prepare_gradient_geometry_inputs(
        delta_x=dx,
        delta_y=dy,
        weights=weights,
        num_dim=4,
        eps_norm=1e-12,
    )
    assert prepared is not None
    scaled_dx, norm_weights = prepared
    expected = tru._low_rank_factor(
        scaled_dx,
        norm_weights,
        dim=4,
        lam_min=1e-4,
        lam_max=1e4,
        eps=1e-6,
        rank_cap=2,
    )
    assert expected is not None
    model.set_gradient_geometry(dx, dy, weights)
    np.testing.assert_allclose(model.low_rank.basis, expected.basis)
    np.testing.assert_allclose(model.low_rank.sqrt_vals, expected.sqrt_vals)
    assert model.low_rank.sqrt_alpha == pytest.approx(expected.sqrt_alpha)


def test_true_ellipsoid_option_b_reclips_after_ema():
    model = tru._TrueEllipsoidGeometryModel(
        num_dim=3,
        metric_sampler="full",
        metric_rank=None,
        update_option="option_b",
        shape_period=1,
        shape_ema=0.5,
        shape_kappa_max=2.0,
    )
    model.ema_cov = np.diag([10.0, 1.0, 1.0]).astype(float)
    dx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [1.1, 0.0, 0.0],
        ],
        dtype=float,
    )
    w = np.array([1.0, 1.0, 1.0], dtype=float)
    model.set_geometry(dx, w)
    assert model.ema_cov is not None
    eigvals = np.linalg.eigvalsh(model.ema_cov)
    assert float(np.max(eigvals) / np.min(eigvals)) <= 2.0 + 1e-6


@pytest.mark.parametrize("metric_sampler", ["full", "low_rank"])
def test_analytic_gradient_matches_single_sample_gradient_update(metric_sampler):
    grad = np.array([1.0, -0.5, 0.25, 0.0], dtype=float)
    analytic = tru._MetricGeometryModel(num_dim=4, metric_sampler=metric_sampler, metric_rank=2)
    generic = tru._MetricGeometryModel(num_dim=4, metric_sampler=metric_sampler, metric_rank=2)
    analytic.set_analytic_gradient(grad)
    generic.set_gradient_geometry(
        grad.reshape(1, -1),
        np.array([np.linalg.norm(grad)], dtype=float),
        np.array([1.0], dtype=float),
    )
    cov_a = analytic.covariance_matrix(jitter=1e-8)
    cov_b = generic.covariance_matrix(jitter=1e-8)
    np.testing.assert_allclose(cov_a, cov_b, rtol=1e-5, atol=1e-7)


def test_cached_mahalanobis_matches_direct_solve():
    model = tru._MetricGeometryModel(num_dim=4, metric_sampler="full", metric_rank=None)
    dx = np.eye(4, dtype=float)
    weights = np.ones((4,), dtype=float)
    model.set_geometry(dx, weights)
    delta = np.array([[0.2, -0.1, 0.3, 0.0]], dtype=float)
    direct = tru._mahalanobis_sq(delta, model.covariance_matrix(jitter=1e-8))
    cached = model.mahalanobis_sq(delta, jitter=1e-8)
    np.testing.assert_allclose(cached, direct, rtol=1e-10, atol=1e-12)


def test_metric_geometry_model_build_step_runtime_checks():
    model = tru._MetricGeometryModel(num_dim=3, metric_sampler="low_rank", metric_rank=2)
    model.low_rank = types.SimpleNamespace(
        sqrt_alpha=0.1,
        basis=np.zeros((2, 2), dtype=float),
        sqrt_vals=np.ones((2,), dtype=float),
    )
    with pytest.raises(RuntimeError):
        model.build_step(np.zeros((1, 3), dtype=float), np.random.default_rng(0))

    model.low_rank = types.SimpleNamespace(
        sqrt_alpha=0.1,
        basis=np.zeros((3, 2), dtype=float),
        sqrt_vals=np.ones((1,), dtype=float),
    )
    with pytest.raises(RuntimeError):
        model.build_step(np.zeros((1, 3), dtype=float), np.random.default_rng(0))


def test_true_ellipsoid_geometry_model_update_and_reset():
    model = tru._TrueEllipsoidGeometryModel(
        num_dim=3,
        metric_sampler="full",
        metric_rank=None,
        update_option="option_b",
        shape_period=1,
        shape_ema=0.5,
    )
    dx = np.array([[1.0, 0.0, 0.0], [0.5, 0.2, 0.1], [-0.3, 0.1, 0.8]], dtype=float)
    w = np.array([1.0, 2.0, 3.0], dtype=float)
    model.set_geometry(dx, w)
    assert model.has_geometry is True
    assert model.ema_cov is not None
    assert model.shape_tick > 0

    model.reset()
    assert model.has_geometry is False
    assert model.ema_cov is None
    assert model.shape_tick == 0


def test_axis_aligned_step_sampler_and_true_ellipsoid_step_sampler(monkeypatch):
    class _Sobol:
        def __init__(self, dim):
            self.dim = dim
            self.calls = 0

        def random(self, n):
            self.calls += 1
            return np.linspace(0.0, 1.0, n * self.dim, dtype=float).reshape(n, self.dim)

    def _fake_raasp(_center, _lb, _ub, num_candidates, *, rng, candidate_rv, sobol_engine, num_pert):
        _ = candidate_rv, sobol_engine, num_pert
        return rng.uniform(-0.5, 0.5, size=(num_candidates, 3))

    monkeypatch.setattr(tru, "generate_raasp_candidates", _fake_raasp)

    axis_sampler = tru._AxisAlignedStepSampler(default_candidate_rv=CandidateRV.SOBOL)
    x_center = np.array([0.5, 0.5, 0.5], dtype=float)
    out = axis_sampler.generate(
        x_center=x_center,
        length=0.5,
        num_dim=3,
        num_candidates=8,
        rng=np.random.default_rng(0),
        candidate_rv=None,
        sobol_engine=None,
        num_pert=2,
        build_step=lambda z, _rng: z,
    )
    assert out.shape == (8, 3)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)

    ell_sampler = tru._TrueEllipsoidStepSampler(
        default_candidate_rv=CandidateRV.SOBOL,
        p_raasp=0.3,
        radial_mode="boundary",
    )
    cov = np.eye(3, dtype=float)
    sobol = _Sobol(dim=8)
    out2 = ell_sampler.generate(
        x_center=np.array([0.8, 0.8, 0.8], dtype=float),
        num_dim=3,
        num_candidates=16,
        length=0.7,
        rng=np.random.default_rng(1),
        candidate_rv=CandidateRV.SOBOL,
        sobol_engine=sobol,
        covariance_matrix=cov,
    )
    assert sobol.calls == 1
    assert out2.shape == (16, 3)
    assert np.all(out2 >= 0.0) and np.all(out2 <= 1.0)


def test_axis_aligned_step_sampler_jax_sobol_fast_path(monkeypatch):
    class _Sobol:
        def random(self, n):
            return np.linspace(0.0, 1.0, n * 3, dtype=float).reshape(n, 3)

    def _boom(*args, **kwargs):
        raise AssertionError("slow ENN generate_raasp_candidates should not run")

    monkeypatch.setattr(tru, "generate_raasp_candidates", _boom)

    axis_sampler = tru._AxisAlignedStepSampler(default_candidate_rv=CandidateRV.SOBOL, use_accel=True)
    out = axis_sampler.generate(
        x_center=np.array([0.5, 0.5, 0.5], dtype=float),
        length=0.5,
        num_dim=3,
        num_candidates=8,
        rng=np.random.default_rng(0),
        candidate_rv=CandidateRV.SOBOL,
        sobol_engine=_Sobol(),
        num_pert=2,
        build_step=lambda z, _rng: z,
        build_candidates=lambda z, _rng, _x_center, _length: z,
    )
    assert out.shape == (8, 3)
    assert np.all(out >= -0.5 - 1e-12)
    assert np.all(out <= 0.5 + 1e-12)


def test_length_policies():
    base = tru._LengthPolicy()
    base.reset()
    base.set_acceptance_ratio(pred=1.0, act=1.0, boundary_hit=False)
    assert base.pending_rho is None
    assert (
        base.apply_after_super_update(
            current_length=1.2,
            base_length=1.0,
            fixed_length=None,
            length_max=2.0,
        )
        == 1.2
    )

    opt = tru._OptionCLengthPolicy(rho_bad=0.25, rho_good=0.75, gamma_down=0.5, gamma_up=2.0)
    opt.set_acceptance_ratio(pred=0.0, act=-1.0, boundary_hit=False)
    shrunk = opt.apply_after_super_update(
        current_length=1.0,
        base_length=1.0,
        fixed_length=None,
        length_max=10.0,
    )
    assert shrunk < 1.0

    opt.set_acceptance_ratio(pred=1.0, act=2.0, boundary_hit=True)
    grown = opt.apply_after_super_update(
        current_length=1.0,
        base_length=1.0,
        fixed_length=None,
        length_max=10.0,
    )
    assert grown > 1.0
    opt.reset()
    assert opt.pending_rho is None
