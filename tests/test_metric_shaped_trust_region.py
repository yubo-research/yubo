import numpy as np
import pytest

from optimizer.trust_region_config import (
    ENNTrueEllipsoidalTrustRegion,
    MetricShapedTRConfig,
    _ray_scale_to_unit_box,
)


def test_metric_shaped_tr_config_length_properties():
    cfg = MetricShapedTRConfig(
        geometry="enn_true_ellipsoid",
        metric_sampler="full",
    )
    assert cfg.length_init > 0
    assert cfg.length_min > 0
    assert cfg.length_max >= cfg.length_min
    assert MetricShapedTRConfig.length_init.fget(cfg) == cfg.length.length_init
    assert MetricShapedTRConfig.length_min.fget(cfg) == cfg.length.length_min
    assert MetricShapedTRConfig.length_max.fget(cfg) == cfg.length.length_max


def _build_true_tr(*, num_dim: int, update_option: str = "option_a", rng_seed: int = 0) -> ENNTrueEllipsoidalTrustRegion:
    cfg = MetricShapedTRConfig(
        geometry="enn_true_ellipsoid",
        metric_sampler="full",
        update_option=update_option,
        p_raasp=0.3,
        radial_mode="ball_uniform",
    )
    rng = np.random.default_rng(rng_seed)
    tr = cfg.build(num_dim=num_dim, rng=rng)
    assert isinstance(tr, ENNTrueEllipsoidalTrustRegion)
    tr.validate_request(num_arms=1)
    return tr


def test_candidates_in_ellipsoid():
    rng = np.random.default_rng(123)
    tr = _build_true_tr(num_dim=7, rng_seed=1)
    delta_x = rng.normal(size=(64, 7))
    weights = np.abs(rng.normal(size=(64,)))
    tr.set_geometry(delta_x, weights)
    x_center = rng.uniform(0.1, 0.9, size=(7,))
    candidates = tr.generate_candidates(
        x_center=x_center,
        lengthscales=None,
        num_candidates=1024,
        rng=rng,
        candidate_rv=tr.candidate_rv,
    )
    cov = tr._covariance_matrix()
    delta = candidates - x_center.reshape(1, -1)
    dist2 = tr._mahalanobis_sq(delta, cov)
    assert np.all(dist2 <= float(tr.length) ** 2 * (1.0 + 1e-6))
    assert np.all(candidates >= -1e-12) and np.all(candidates <= 1.0 + 1e-12)


def test_ray_scaling_preserves_ellipsoid():
    tr = _build_true_tr(num_dim=5, rng_seed=2)
    cov = tr._covariance_matrix()
    x_center = np.full((5,), 0.95, dtype=float)
    step = np.zeros((1, 5), dtype=float)
    step[0, 0] = 0.9 * float(tr.length)
    x_inside = x_center.reshape(1, -1) + step
    x_scaled = _ray_scale_to_unit_box(x_center, x_inside)
    delta = x_scaled - x_center.reshape(1, -1)
    dist2 = tr._mahalanobis_sq(delta, cov)
    assert float(dist2[0]) <= float(tr.length) ** 2 * (1.0 + 1e-8)
    assert np.all(x_scaled >= 0.0) and np.all(x_scaled <= 1.0)


def test_metric_matrix_is_spd():
    rng = np.random.default_rng(42)
    tr = _build_true_tr(num_dim=6, rng_seed=3)
    tr.set_geometry(rng.normal(size=(80, 6)), np.abs(rng.normal(size=(80,))))
    cov = tr._covariance_matrix()
    np.linalg.cholesky(cov)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0.0)


def test_option_a_length_expands_then_shrinks():
    tr = _build_true_tr(num_dim=4, update_option="option_a", rng_seed=4)
    l0 = float(tr.length)
    tr.update(np.array([0.0]), np.array([0.0]))
    n = 1
    for i in range(3):
        n += 1
        y_obs = np.arange(n, dtype=float)
        tr.update(y_obs, np.array([float(10 + i)]))
    l_up = float(tr.length)
    assert l_up >= l0
    ft = tr.failure_tolerance
    best = float(tr.best_value)
    for _ in range(ft):
        n += 1
        tr.update(np.zeros((n,), dtype=float), np.array([best]))
    assert float(tr.length) < l_up


def test_option_c_rho_controller():
    tr = _build_true_tr(num_dim=4, update_option="option_c", rng_seed=5)
    l0 = float(tr.length)
    tr.set_acceptance_ratio(pred=1.0, act=-1.0, boundary_hit=False)
    tr.update(np.array([0.0]), np.array([0.0]))
    l1 = float(tr.length)
    assert l1 < l0
    tr.set_acceptance_ratio(pred=1.0, act=2.0, boundary_hit=True)
    tr.update(np.array([0.0, 0.0]), np.array([0.0]))
    assert float(tr.length) > l1


def test_option_c_keeps_base_turbo_bookkeeping_progressing():
    tr = _build_true_tr(num_dim=4, update_option="option_c", rng_seed=6)
    for i in range(3):
        if i > 0:
            tr.set_acceptance_ratio(pred=1.0, act=(-1.0 if i % 2 else 1.0), boundary_hit=True)
        y_obs = np.arange(i + 1, dtype=float)
        tr.update(y_obs, np.array([float(i)]))
    assert int(tr.prev_num_obs) == 3
    assert float(tr.best_value) == 2.0


def test_option_c_bad_rho_can_trigger_restart():
    tr = _build_true_tr(num_dim=4, update_option="option_c", rng_seed=7)
    n = 1
    tr.update(np.array([0.0]), np.array([0.0]))
    for _ in range(64):
        tr.set_acceptance_ratio(pred=1.0, act=-1.0, boundary_hit=False)
        n += 1
        y_obs = np.arange(n, dtype=float)
        tr.update(y_obs, np.array([0.0]))
        if tr.needs_restart():
            break
    assert tr.needs_restart()


def test_boundary_tol_rejects_values_above_one():
    with pytest.raises(ValueError, match=r"'boundary_tol' must be <= 1"):
        MetricShapedTRConfig(
            geometry="enn_true_ellipsoid",
            metric_sampler="full",
            boundary_tol=1.1,
        )


def test_record_incumbent_transition_tracks_previous_state():
    tr = _build_true_tr(num_dim=3, rng_seed=11)
    x0 = np.array([0.1, 0.2, 0.3], dtype=float)
    x1 = np.array([0.2, 0.2, 0.3], dtype=float)
    assert tr.record_incumbent_transition(x_center=x0, y_value=1.0) is None
    prev = tr.record_incumbent_transition(x_center=x1, y_value=2.0)
    assert prev is not None
    prev_val, prev_x = prev
    assert float(prev_val) == 1.0
    assert np.allclose(prev_x, x0)


def test_set_analytic_gradient_geometry():
    cfg = MetricShapedTRConfig(
        geometry="enn_grad_metric_shaped",
        metric_sampler="full",
        update_option="option_a",
    )
    tr = cfg.build(num_dim=3, rng=np.random.default_rng(31))
    tr.validate_request(num_arms=1)
    assert not tr.has_geometry
    tr.set_analytic_gradient_geometry(np.array([1.0, -0.5, 0.0]))
    assert tr.has_geometry
    cov = tr._covariance_matrix()
    np.linalg.cholesky(cov)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0.0)


def test_needs_gradient_signal_by_geometry():
    cfg_grad = MetricShapedTRConfig(
        geometry="enn_grad_true_ellipsoid",
        update_option="option_a",
    )
    tr_grad = cfg_grad.build(num_dim=3, rng=np.random.default_rng(21))
    assert isinstance(tr_grad, ENNTrueEllipsoidalTrustRegion)
    assert tr_grad.needs_gradient_signal() is True

    cfg_plain = MetricShapedTRConfig(
        geometry="enn_true_ellipsoid",
        update_option="option_a",
    )
    tr_plain = cfg_plain.build(num_dim=3, rng=np.random.default_rng(22))
    assert isinstance(tr_plain, ENNTrueEllipsoidalTrustRegion)
    assert tr_plain.needs_gradient_signal() is False


def test_pc_rotation_geometry_full():
    rng = np.random.default_rng(99)
    cfg = MetricShapedTRConfig(
        geometry="enn_true_ellipsoid",
        metric_sampler="low_rank",
        pc_rotation_mode="full",
    )
    tr = cfg.build(num_dim=5, rng=rng)
    x_center = np.full(5, 0.5)
    x_obs = x_center + rng.normal(0, 0.05, size=(30, 5))
    y_obs = rng.uniform(0, 1, size=30)
    tr.observe_pc_rotation_geometry(x_center=x_center, x_obs=x_obs, y_obs=y_obs, maximize=True)
    assert tr.has_geometry
    cov = tr._covariance_matrix()
    np.linalg.cholesky(cov)
    candidates = tr.generate_candidates(
        x_center=x_center,
        lengthscales=None,
        num_candidates=32,
        rng=rng,
        candidate_rv=tr.candidate_rv,
    )
    assert candidates.shape == (32, 5)
    assert np.all(candidates >= -1e-10) and np.all(candidates <= 1.0 + 1e-10)


def test_pc_rotation_geometry_low_rank():
    rng = np.random.default_rng(100)
    cfg = MetricShapedTRConfig(
        geometry="enn_true_ellipsoid",
        metric_sampler="low_rank",
        pc_rotation_mode="low_rank",
        pc_rank=3,
    )
    tr = cfg.build(num_dim=8, rng=rng)
    x_center = np.full(8, 0.5)
    x_obs = x_center + rng.normal(0, 0.03, size=(50, 8))
    y_obs = rng.uniform(0, 1, size=50)
    tr.observe_pc_rotation_geometry(x_center=x_center, x_obs=x_obs, y_obs=y_obs, maximize=True)
    assert tr.has_geometry
    candidates = tr.generate_candidates(
        x_center=x_center,
        lengthscales=None,
        num_candidates=16,
        rng=rng,
        candidate_rv=tr.candidate_rv,
    )
    assert candidates.shape == (16, 8)
