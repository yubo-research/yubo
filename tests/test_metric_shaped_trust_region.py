import types

import numpy as np
import pytest
import torch
from enn.turbo.config.candidate_rv import CandidateRV

import optimizer.trust_region_utils as tr_utils
from optimizer.box_trust_region import (
    FixedLengthTurboTrustRegion,
    ModuleAwareTrustRegion,
    maybe_enable_module_masks,
)
from optimizer.metric_trust_region import ENNMetricShapedTrustRegion, MetricShapedTrustRegion
from optimizer.trust_region_config import (
    ENNGradientIsotropicTrustRegion,
    ENNIsotropicTrustRegion,
    ENNTrueEllipsoidalTrustRegion,
    MetricShapedTRConfig,
    _ray_scale_to_unit_box,
)
from optimizer.trust_region_math import _LowRankFactor
from optimizer.trust_region_sampling_utils import _low_rank_mahalanobis_sq


def test_metric_shaped_tr_config_length_properties():
    cfg = MetricShapedTRConfig(
        geometry="enn_ellip",
        metric_sampler="full",
    )
    assert cfg.length_init > 0
    assert cfg.length_min > 0
    assert cfg.length_max >= cfg.length_min
    assert MetricShapedTRConfig.length_init.fget(cfg) == cfg.length.length_init
    assert MetricShapedTRConfig.length_min.fget(cfg) == cfg.length.length_min
    assert MetricShapedTRConfig.length_max.fget(cfg) == cfg.length.length_max


def test_geometry_aliases_canonicalize_to_short_names():
    cfg_metric = MetricShapedTRConfig(geometry="enn_metr", metric_sampler="full")
    cfg_iso = MetricShapedTRConfig(geometry="enn_iso")
    cfg_grad_iso = MetricShapedTRConfig(geometry="grad_iso")
    cfg_ellip = MetricShapedTRConfig(geometry="enn_ellip", metric_sampler="full")
    cfg_grad_metric = MetricShapedTRConfig(geometry="grad_metr", metric_sampler="full")
    cfg_grad_ellip = MetricShapedTRConfig(geometry="grad_ellip", metric_sampler="full")

    assert cfg_metric.geometry == "enn_metr"
    assert cfg_iso.geometry == "enn_iso"
    assert cfg_grad_iso.geometry == "grad_iso"
    assert cfg_ellip.geometry == "enn_ellip"
    assert cfg_grad_metric.geometry == "grad_metr"
    assert cfg_grad_ellip.geometry == "grad_ellip"


def test_identity_geometries_reject_full_sampler_and_rank():
    with pytest.raises(ValueError, match="only supports sampler='low_rank'"):
        MetricShapedTRConfig(geometry="enn_iso", metric_sampler="full")
    with pytest.raises(ValueError, match="does not use metric_rank"):
        MetricShapedTRConfig(geometry="enn_iso", metric_rank=2)
    with pytest.raises(ValueError, match="only supports sampler='low_rank'"):
        MetricShapedTRConfig(geometry="grad_iso", metric_sampler="full")
    with pytest.raises(ValueError, match="does not use metric_rank"):
        MetricShapedTRConfig(geometry="grad_iso", metric_rank=2)


def test_identity_metric_geometry_stays_isotropic_and_fixed():
    rng = np.random.default_rng(11)
    cfg = MetricShapedTRConfig(
        geometry="enn_iso",
        p_raasp=0.3,
        radial_mode="ball_uniform",
    )
    tr = cfg.build(num_dim=6, rng=rng)
    assert isinstance(tr, ENNIsotropicTrustRegion)
    assert tr._geometry_model.metric_sampler == "low_rank"
    assert tr._geometry_model.low_rank.basis.shape == (6, 0)
    assert float(tr._geometry_model.low_rank.sqrt_alpha) == 1.0
    cov0 = tr._covariance_matrix()
    assert np.allclose(cov0, np.eye(6) * float(cov0[0, 0]))
    tr.set_geometry(rng.normal(size=(64, 6)), np.abs(rng.normal(size=(64,))))
    cov1 = tr._covariance_matrix()
    np.testing.assert_allclose(cov1, cov0)
    x_center = rng.uniform(0.2, 0.8, size=(6,))
    candidates = tr.generate_candidates(
        x_center=x_center,
        lengthscales=None,
        num_candidates=256,
        rng=rng,
        candidate_rv=CandidateRV.UNIFORM,
    )
    delta = candidates - x_center.reshape(1, -1)
    dist2 = tr._mahalanobis_sq(delta, cov1)
    assert np.all(dist2 <= float(tr.length) ** 2 * (1.0 + 1e-6))


def test_gradient_identity_geometry_uses_gradient_only_for_sampling(monkeypatch):
    rng = np.random.default_rng(17)
    cfg = MetricShapedTRConfig(
        geometry="grad_iso",
        p_raasp=0.3,
        radial_mode="ball_uniform",
    )
    tr = cfg.build(num_dim=6, rng=rng)
    assert isinstance(tr, ENNGradientIsotropicTrustRegion)
    assert tr.needs_gradient_signal() is True
    assert tr._geometry_model.metric_sampler == "low_rank"
    cov0 = tr._covariance_matrix()
    assert np.allclose(cov0, np.eye(6) * float(cov0[0, 0]))

    captured = {}

    def fake_generate(self, **kwargs):
        captured["num_pert"] = int(kwargs["num_pert"])
        x_center = np.asarray(kwargs["x_center"], dtype=float).reshape(1, -1)
        return np.repeat(x_center, int(kwargs["num_candidates"]), axis=0)

    monkeypatch.setattr(type(tr._step_sampler), "generate", fake_generate)

    x_center = rng.uniform(0.2, 0.8, size=(6,))
    tr.generate_candidates(
        x_center=x_center,
        lengthscales=None,
        num_candidates=32,
        rng=rng,
        candidate_rv=CandidateRV.UNIFORM,
    )
    base_num_pert = captured["num_pert"]

    tr.set_analytic_gradient_geometry(np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    cov1 = tr._covariance_matrix()
    np.testing.assert_allclose(cov1, cov0)
    tr.generate_candidates(
        x_center=x_center,
        lengthscales=None,
        num_candidates=32,
        rng=rng,
        candidate_rv=CandidateRV.UNIFORM,
    )
    assert captured["num_pert"] < base_num_pert
    reduced_num_pert = captured["num_pert"]
    assert reduced_num_pert < base_num_pert
    tr.generate_candidates(
        x_center=x_center,
        lengthscales=None,
        num_candidates=32,
        rng=rng,
        candidate_rv=CandidateRV.UNIFORM,
    )
    assert captured["num_pert"] == base_num_pert


def _build_true_tr(*, num_dim: int, update_option: str = "option_a", rng_seed: int = 0) -> ENNTrueEllipsoidalTrustRegion:
    cfg = MetricShapedTRConfig(
        geometry="enn_ellip",
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
        candidate_rv=CandidateRV.UNIFORM,
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


def test_full_metric_mahalanobis_uses_factor_and_avoids_inverse(monkeypatch):
    tr = _build_true_tr(num_dim=4, rng_seed=18)
    tr.set_geometry(
        delta_x=np.array(
            [
                [0.2, 0.0, 0.0, 0.0],
                [0.0, 0.3, 0.0, 0.0],
                [0.0, 0.0, 0.4, 0.0],
                [0.0, 0.0, 0.0, 0.5],
            ],
            dtype=float,
        ),
        weights=np.array([0.1, 0.2, 0.3, 0.4], dtype=float),
    )
    monkeypatch.setattr(
        tr._geometry_model,
        "covariance_inverse",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("inverse should not be used")),
    )
    delta = np.array([[0.1, 0.0, 0.0, 0.0]], dtype=float)
    out = tr._mahalanobis_sq(delta, np.eye(4))
    assert out.shape == (1,)
    assert float(out[0]) > 0.0


def test_use_jax_full_metric_mahalanobis_routes_through_factor_helper(monkeypatch):
    cfg = MetricShapedTRConfig(
        geometry="enn_metr",
        metric_sampler="full",
        use_accel=True,
    )
    rng = np.random.default_rng(19)
    tr = cfg.build(num_dim=4, rng=rng)
    tr.set_geometry(
        delta_x=np.array(
            [
                [0.2, 0.0, 0.0, 0.0],
                [0.0, 0.3, 0.0, 0.0],
                [0.0, 0.0, 0.4, 0.0],
                [0.0, 0.0, 0.0, 0.5],
            ],
            dtype=float,
        ),
        weights=np.array([0.1, 0.2, 0.3, 0.4], dtype=float),
    )
    calls = {}

    def _fake_factor(delta, factor):
        calls["delta"] = np.asarray(delta, dtype=float)
        calls["factor"] = np.asarray(factor, dtype=float)
        return np.full((np.asarray(delta).shape[0],), 1.0, dtype=float)

    monkeypatch.setattr("optimizer.metric_trust_region._accel.mahalanobis_sq_from_factor", _fake_factor)
    delta = np.array([[0.1, 0.0, 0.0, 0.0]], dtype=float)
    out = tr._mahalanobis_sq(delta, np.eye(4))
    assert out.shape == (1,)
    assert "factor" in calls
    assert calls["factor"].shape == (4, 4)


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
            geometry="enn_ellip",
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


def test_observe_incumbent_transition_uses_geometry_mahalanobis(monkeypatch):
    tr = _build_true_tr(num_dim=3, rng_seed=12)
    tr.record_incumbent_transition(x_center=np.array([0.1, 0.2, 0.3], dtype=float), y_value=1.0)
    calls = {}

    def _fake_mahal(delta, *, jitter):
        calls["delta"] = np.asarray(delta, dtype=float)
        calls["jitter"] = float(jitter)
        return np.array([float(tr.length) ** 2], dtype=float)

    monkeypatch.setattr(tr._geometry_model, "mahalanobis_sq", _fake_mahal)
    monkeypatch.setattr(
        tr,
        "set_acceptance_ratio",
        lambda *, pred, act, boundary_hit: calls.update(
            {
                "pred": float(pred),
                "act": float(act),
                "boundary_hit": bool(boundary_hit),
            }
        ),
    )

    tr.observe_incumbent_transition(
        x_center=np.array([0.2, 0.2, 0.3], dtype=float),
        y_value=2.0,
        predict_delta=lambda _prev, _curr: 1.0,
    )

    assert calls["delta"].shape == (1, 3)
    assert calls["jitter"] == pytest.approx(float(getattr(tr.config, "shape_jitter", 1e-6)))
    assert calls["pred"] == pytest.approx(1.0)
    assert calls["act"] == pytest.approx(1.0)
    assert calls["boundary_hit"] is True


def test_low_rank_mahalanobis_matches_dense():
    rng = np.random.default_rng(13)
    q, _ = np.linalg.qr(rng.normal(size=(3, 2)))
    low_rank = _LowRankFactor(
        sqrt_alpha=0.6,
        basis=q,
        sqrt_vals=np.array([1.4, 0.8], dtype=float),
    )
    delta = rng.normal(size=(8, 3))
    alpha = float(low_rank.sqrt_alpha) ** 2
    lam = np.square(low_rank.sqrt_vals)
    cov = alpha * np.eye(3, dtype=float) + (low_rank.basis * lam.reshape(1, -1)) @ low_rank.basis.T
    dense = np.einsum("nd,de,ne->n", delta, np.linalg.inv(cov), delta)
    low_rank_sq = _low_rank_mahalanobis_sq(delta, low_rank, use_accel=False)
    assert np.allclose(low_rank_sq, dense)


def test_low_rank_ellipsoid_transition_avoids_dense_covariance(monkeypatch):
    cfg = MetricShapedTRConfig(
        geometry="enn_ellip",
        metric_sampler="low_rank",
        metric_rank=2,
        update_option="option_a",
    )
    tr = cfg.build(num_dim=3, rng=np.random.default_rng(14))
    tr.validate_request(num_arms=1)
    tr.record_incumbent_transition(
        x_center=np.array([0.1, 0.2, 0.3], dtype=float),
        y_value=1.0,
    )

    calls = {}

    def _fake_low_rank(delta, low_rank, *, use_accel):
        calls["delta"] = np.asarray(delta, dtype=float)
        calls["use_accel"] = bool(use_accel)
        return np.array([float(tr.length) ** 2], dtype=float)

    monkeypatch.setattr("optimizer.ellipsoidal_trust_region._low_rank_mahalanobis_sq", _fake_low_rank)
    monkeypatch.setattr(tr._geometry_model, "covariance_matrix", lambda **kwargs: (_ for _ in ()).throw(AssertionError("dense covariance should not be used")))
    monkeypatch.setattr(
        tr,
        "set_acceptance_ratio",
        lambda *, pred, act, boundary_hit: calls.update(
            {
                "pred": float(pred),
                "act": float(act),
                "boundary_hit": bool(boundary_hit),
            }
        ),
    )

    tr.observe_incumbent_transition(
        x_center=np.array([0.2, 0.2, 0.3], dtype=float),
        y_value=2.0,
        predict_delta=lambda _prev, _curr: 1.0,
    )

    assert calls["delta"].shape == (1, 3)
    assert calls["use_accel"] is False
    assert calls["pred"] == pytest.approx(1.0)
    assert calls["act"] == pytest.approx(1.0)
    assert calls["boundary_hit"] is True


def test_low_rank_ellipsoid_candidates_skip_dense_covariance(monkeypatch):
    cfg = MetricShapedTRConfig(
        geometry="enn_ellip",
        metric_sampler="low_rank",
        metric_rank=2,
        update_option="option_a",
    )
    tr = cfg.build(num_dim=4, rng=np.random.default_rng(15))
    tr.validate_request(num_arms=1)
    monkeypatch.setattr(
        tr,
        "_covariance_matrix",
        lambda: (_ for _ in ()).throw(AssertionError("dense covariance should not be used")),
    )

    candidates = tr.generate_candidates(
        x_center=np.array([0.25, 0.25, 0.25, 0.25], dtype=float),
        lengthscales=None,
        num_candidates=16,
        rng=np.random.default_rng(16),
        candidate_rv=CandidateRV.UNIFORM,
    )

    assert candidates.shape == (16, 4)
    assert np.all(candidates >= -1e-12)
    assert np.all(candidates <= 1.0 + 1e-12)


def test_full_ellipsoid_path_avoids_dense_covariance(monkeypatch):
    cfg = MetricShapedTRConfig(
        geometry="enn_ellip",
        metric_sampler="full",
        update_option="option_a",
    )
    tr = cfg.build(num_dim=4, rng=np.random.default_rng(19))
    tr.validate_request(num_arms=1)
    tr.record_incumbent_transition(
        x_center=np.array([0.15, 0.2, 0.25, 0.3], dtype=float),
        y_value=1.0,
    )
    tr.set_geometry(
        delta_x=np.array(
            [
                [0.2, 0.0, 0.0, 0.0],
                [0.0, 0.3, 0.0, 0.0],
                [0.0, 0.0, 0.4, 0.0],
                [0.0, 0.0, 0.0, 0.5],
            ],
            dtype=float,
        ),
        weights=np.array([0.1, 0.2, 0.3, 0.4], dtype=float),
    )

    monkeypatch.setattr(
        tr._geometry_model,
        "covariance_matrix",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("dense covariance should not be used")),
    )
    monkeypatch.setattr(
        tr._geometry_model,
        "covariance_inverse",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("dense inverse should not be used")),
    )
    monkeypatch.setattr(tr, "set_acceptance_ratio", lambda *, pred, act, boundary_hit: None)

    tr.observe_incumbent_transition(
        x_center=np.array([0.2, 0.2, 0.25, 0.3], dtype=float),
        y_value=2.0,
        predict_delta=lambda _prev, _curr: 1.0,
    )

    candidates = tr.generate_candidates(
        x_center=np.array([0.2, 0.2, 0.25, 0.3], dtype=float),
        lengthscales=None,
        num_candidates=16,
        rng=np.random.default_rng(20),
        candidate_rv=CandidateRV.UNIFORM,
    )
    assert candidates.shape == (16, 4)
    assert np.all(candidates >= -1e-12)
    assert np.all(candidates <= 1.0 + 1e-12)


def test_low_rank_metric_mahalanobis_avoids_dense_covariance(monkeypatch):
    cfg = MetricShapedTRConfig(
        geometry="enn_metr",
        metric_sampler="low_rank",
        metric_rank=2,
    )
    tr = cfg.build(num_dim=4, rng=np.random.default_rng(17))
    tr.validate_request(num_arms=1)
    tr.set_geometry(
        delta_x=np.array(
            [
                [0.05, 0.0, 0.0, 0.0],
                [0.0, 0.04, 0.0, 0.0],
                [0.0, 0.0, 0.03, 0.0],
                [0.0, 0.0, 0.0, 0.02],
            ],
            dtype=float,
        ),
        weights=np.array([0.4, 0.3, 0.2, 0.1], dtype=float),
    )

    calls = {}

    def _fake_low_rank(delta, low_rank, *, use_accel):
        calls["delta"] = np.asarray(delta, dtype=float)
        calls["use_accel"] = bool(use_accel)
        return np.array([3.25], dtype=float)

    monkeypatch.setattr(tr_utils, "_low_rank_mahalanobis_sq", _fake_low_rank)
    monkeypatch.setattr(
        tr._geometry_model,
        "covariance_matrix",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("dense covariance should not be used")),
    )

    out = tr._mahalanobis_sq(np.array([[0.1, 0.0, 0.0, 0.0]], dtype=float), np.eye(4))

    assert np.allclose(out, np.array([3.25], dtype=float))
    assert calls["delta"].shape == (1, 4)
    assert calls["use_accel"] is False


def test_set_analytic_gradient_geometry():
    cfg = MetricShapedTRConfig(
        geometry="grad_metr",
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
        geometry="grad_ellip",
        update_option="option_a",
    )
    tr_grad = cfg_grad.build(num_dim=3, rng=np.random.default_rng(21))
    assert isinstance(tr_grad, ENNTrueEllipsoidalTrustRegion)
    assert tr_grad.needs_gradient_signal() is True

    cfg_plain = MetricShapedTRConfig(
        geometry="enn_ellip",
        update_option="option_a",
    )
    tr_plain = cfg_plain.build(num_dim=3, rng=np.random.default_rng(22))
    assert isinstance(tr_plain, ENNTrueEllipsoidalTrustRegion)
    assert tr_plain.needs_gradient_signal() is False


def test_pc_rotation_geometry_full():
    rng = np.random.default_rng(99)
    cfg = MetricShapedTRConfig(
        geometry="enn_ellip",
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
        candidate_rv=CandidateRV.UNIFORM,
    )
    assert candidates.shape == (32, 5)


def test_module_aware_box_trust_region_uses_leaf_blocks(monkeypatch):
    policy = torch.nn.Sequential(
        torch.nn.LayerNorm(4),
        torch.nn.Linear(4, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 2),
    )
    policy.num_params = lambda: sum(p.numel() for p in policy.parameters())

    cfg = MetricShapedTRConfig(geometry="box")
    base_tr = FixedLengthTurboTrustRegion(config=cfg, num_dim=policy.num_params())
    opt = types.SimpleNamespace(_tr_state=base_tr)

    enabled = maybe_enable_module_masks(opt, policy, enabled=True, min_num_params=0, block_prob=0.5)
    assert enabled is True
    assert isinstance(opt._tr_state, ModuleAwareTrustRegion)

    tr = opt._tr_state
    x_center = np.full((policy.num_params(),), 0.5, dtype=float)
    rng = np.random.default_rng(0)
    candidates = tr.generate_candidates(
        x_center=x_center,
        lengthscales=None,
        num_candidates=64,
        rng=rng,
        candidate_rv=CandidateRV.UNIFORM,
    )
    assert candidates.shape == (64, policy.num_params())
    assert np.all(candidates >= 0.0) and np.all(candidates <= 1.0)

    diff = np.abs(candidates - x_center.reshape(1, -1)) > 1e-12
    for row in diff:
        for start, end in tr.block_slices:
            block = row[start:end]
            if block.any():
                assert block.all()
    assert np.all(candidates >= -1e-10) and np.all(candidates <= 1.0 + 1e-10)


def test_module_aware_box_trust_region_preserves_candidate_rv(monkeypatch):
    policy = torch.nn.Sequential(
        torch.nn.LayerNorm(4),
        torch.nn.Linear(4, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 2),
    )
    policy.num_params = lambda: sum(p.numel() for p in policy.parameters())

    cfg = MetricShapedTRConfig(geometry="box")
    base_tr = FixedLengthTurboTrustRegion(config=cfg, num_dim=policy.num_params())
    opt = types.SimpleNamespace(_tr_state=base_tr)

    enabled = maybe_enable_module_masks(opt, policy, enabled=True, min_num_params=0, block_prob=0.5)
    assert enabled is True

    calls = {}

    def fake_generate(center, lb, ub, num_candidates, *, rng, candidate_rv, sobol_engine, block_slices, block_prob):
        calls["candidate_rv"] = candidate_rv
        return np.tile(np.asarray(center, dtype=float).reshape(1, -1), (int(num_candidates), 1))

    monkeypatch.setattr("optimizer.box_trust_region._generate_block_raasp_candidates", fake_generate)
    tr = opt._tr_state
    tr.generate_candidates(
        x_center=np.full((policy.num_params(),), 0.5, dtype=float),
        lengthscales=None,
        num_candidates=8,
        rng=np.random.default_rng(0),
        candidate_rv=CandidateRV.SOBOL,
    )
    assert calls["candidate_rv"] == CandidateRV.SOBOL


def test_module_aware_metric_trust_region_reports_enabled_without_replacement():
    policy = torch.nn.Sequential(
        torch.nn.LayerNorm(4),
        torch.nn.Linear(4, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 2),
    )
    policy.num_params = lambda: sum(p.numel() for p in policy.parameters())

    cfg = MetricShapedTRConfig(geometry="enn_metr")
    base_tr = ENNMetricShapedTrustRegion(config=cfg, num_dim=policy.num_params())
    base_tr.module_block_slices = ((0, 2), (2, 4))
    opt = types.SimpleNamespace(_tr_state=base_tr)

    enabled = maybe_enable_module_masks(opt, policy, enabled=True, min_num_params=0, block_prob=0.5)
    assert enabled is True
    assert opt._tr_state is base_tr


def test_module_aware_box_trust_region_is_disabled_by_default():
    policy = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    policy.num_params = lambda: sum(p.numel() for p in policy.parameters())

    cfg = MetricShapedTRConfig(geometry="box")
    base_tr = FixedLengthTurboTrustRegion(config=cfg, num_dim=policy.num_params())
    opt = types.SimpleNamespace(_tr_state=base_tr)

    enabled = maybe_enable_module_masks(opt, policy, min_num_params=0, block_prob=0.5)
    assert enabled is False
    assert opt._tr_state is base_tr


def test_module_aware_box_trust_region_can_be_enabled_explicitly():
    policy = torch.nn.Sequential(
        torch.nn.LayerNorm(4),
        torch.nn.Linear(4, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 2),
    )
    policy.num_params = lambda: sum(p.numel() for p in policy.parameters())

    cfg = MetricShapedTRConfig(geometry="box")
    base_tr = FixedLengthTurboTrustRegion(config=cfg, num_dim=policy.num_params())
    opt = types.SimpleNamespace(_tr_state=base_tr)

    enabled = maybe_enable_module_masks(
        opt,
        policy,
        enabled=True,
        min_num_params=0,
        block_prob=0.5,
    )
    assert enabled is True
    assert isinstance(opt._tr_state, ModuleAwareTrustRegion)


def test_metric_trust_region_rejects_non_default_radial_mode():
    with pytest.raises(ValueError, match="radial_mode='ball_uniform'"):
        MetricShapedTRConfig(geometry="enn_metr", radial_mode="boundary")


def test_module_blocks_apply_to_metric_candidate_generation(monkeypatch):
    rng = np.random.default_rng(7)
    cfg = MetricShapedTRConfig(
        geometry="enn_metr",
        metric_sampler="low_rank",
    )
    tr = cfg.build(num_dim=6, rng=rng)
    tr.module_block_slices = ((0, 2), (2, 4), (4, 6))
    tr.module_block_prob = 1.0
    calls = []

    def fake_apply(candidates, *, rng, candidate_rv, sobol_engine, block_slices, block_prob):
        calls.append((candidates.shape, tuple(block_slices), block_prob, candidate_rv))
        return candidates

    monkeypatch.setattr("optimizer.metric_trust_region._apply_block_raasp_mask", fake_apply)
    x_center = np.full(6, 0.5)
    candidates = tr.generate_candidates(
        x_center=x_center,
        lengthscales=None,
        num_candidates=8,
        rng=rng,
        candidate_rv=CandidateRV.UNIFORM,
    )
    assert candidates.shape == (8, 6)
    assert calls and calls[0][1] == ((0, 2), (2, 4), (4, 6))


def test_pc_rotation_geometry_low_rank():
    rng = np.random.default_rng(100)
    cfg = MetricShapedTRConfig(
        geometry="enn_ellip",
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
        candidate_rv=CandidateRV.UNIFORM,
    )
    assert candidates.shape == (16, 8)


def test_low_rank_geometry_skips_full_covariance_build(monkeypatch):
    model = tr_utils._MetricGeometryModel(
        num_dim=4,
        metric_sampler="low_rank",
        metric_rank=2,
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
    weights = np.ones((4,), dtype=float)

    def _fail_trace_normalize(*args, **kwargs):
        raise AssertionError("low-rank path should not build full covariance")

    monkeypatch.setattr(tr_utils, "_trace_normalize", _fail_trace_normalize)

    model.set_geometry(dx, weights)
    assert model.has_geometry
    assert model.metric_sampler == "low_rank"

    model.reset()
    model.set_gradient_geometry(
        dx,
        np.array([1.0, 0.8, 0.6, 0.4], dtype=float),
        weights,
    )
    assert model.has_geometry
    assert model.metric_sampler == "low_rank"


def test_box_trust_region_fixed_length_update_and_restart():
    cfg = MetricShapedTRConfig(
        geometry="box",
        fixed_length=0.77,
    )
    tr = cfg.build(num_dim=3, rng=np.random.default_rng(123))
    tr.validate_request(num_arms=1)
    assert float(tr.length) == pytest.approx(0.77, abs=1e-12)

    tr.update(np.array([0.0]), np.array([0.0]))
    assert float(tr.length) == pytest.approx(0.77, abs=1e-12)
    tr.restart(rng=np.random.default_rng(9))
    assert float(tr.length) == pytest.approx(0.77, abs=1e-12)


def test_metric_shaped_generate_candidates_invalid_inputs():
    cfg = MetricShapedTRConfig(
        geometry="enn_metr",
        metric_sampler="full",
    )
    tr = cfg.build(num_dim=3, rng=np.random.default_rng(44))
    tr.validate_request(num_arms=1)
    tr.set_geometry(np.eye(3, dtype=float), np.ones((3,), dtype=float))
    x_center = np.array([0.5, 0.5, 0.5], dtype=float)

    with pytest.raises(ValueError, match="lengthscales are not supported"):
        tr.generate_candidates(
            x_center=x_center,
            lengthscales=np.ones((3,), dtype=float),
            num_candidates=4,
            rng=np.random.default_rng(1),
        )
    with pytest.raises(ValueError):
        tr.generate_candidates(
            x_center=x_center,
            lengthscales=None,
            num_candidates=0,
            rng=np.random.default_rng(1),
        )
    with pytest.raises(ValueError):
        tr.generate_candidates(
            x_center=np.array([0.5, 0.5], dtype=float),
            lengthscales=None,
            num_candidates=4,
            rng=np.random.default_rng(1),
        )


def test_metric_and_box_class_methods_are_directly_covered():
    metric_cfg = MetricShapedTRConfig(
        geometry="enn_metr",
        metric_sampler="full",
    )
    metric_tr = metric_cfg.build(num_dim=3, rng=np.random.default_rng(55))
    assert isinstance(metric_tr, ENNMetricShapedTrustRegion)
    metric_tr.validate_request(num_arms=1)
    MetricShapedTrustRegion.set_geometry(
        metric_tr,
        np.eye(3, dtype=float),
        np.ones((3,), dtype=float),
    )
    ENNMetricShapedTrustRegion.set_gradient_geometry(
        metric_tr,
        np.eye(3, dtype=float),
        np.ones((3,), dtype=float),
        np.ones((3,), dtype=float),
    )
    MetricShapedTrustRegion.set_acceptance_ratio(metric_tr, pred=1.0, act=1.0, boundary_hit=False)
    MetricShapedTrustRegion.update(metric_tr, np.array([0.0]), np.array([0.0]))
    MetricShapedTrustRegion.restart(metric_tr, rng=np.random.default_rng(56))

    box_cfg = MetricShapedTRConfig(
        geometry="box",
        fixed_length=0.66,
    )
    box_tr = box_cfg.build(num_dim=2, rng=np.random.default_rng(57))
    assert isinstance(box_tr, FixedLengthTurboTrustRegion)
    box_tr.validate_request(num_arms=1)
    FixedLengthTurboTrustRegion.update(box_tr, np.array([0.0]), np.array([0.0]))
    FixedLengthTurboTrustRegion.restart(box_tr, rng=np.random.default_rng(58))
