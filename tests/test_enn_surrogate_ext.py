import types

import numpy as np
import pytest
from enn.turbo.config.enn_fit_config import ENNFitConfig
from enn.turbo.config.enn_surrogate_config import ENNSurrogateConfig

from optimizer.enn_surrogate_ext import GeometryENNSurrogate
from optimizer.trust_region_config import (
    ENNTrueEllipsoidalTrustRegion,
    MetricShapedTRConfig,
)


def _build_true_tr(*, num_dim: int, rng_seed: int = 0) -> ENNTrueEllipsoidalTrustRegion:
    cfg = MetricShapedTRConfig(
        geometry="enn_true_ellipsoid",
        metric_sampler="full",
        update_option="option_c",
        p_raasp=0.3,
        radial_mode="ball_uniform",
    )
    rng = np.random.default_rng(rng_seed)
    tr = cfg.build(num_dim=num_dim, rng=rng)
    assert isinstance(tr, ENNTrueEllipsoidalTrustRegion)
    return tr


class _Posterior:
    def __init__(self, mu):
        self.mu = np.asarray(mu, dtype=float)


def _make_surrogate(predict_fn) -> GeometryENNSurrogate:
    surrogate = object.__new__(GeometryENNSurrogate)
    surrogate.predict = types.MethodType(lambda self, x: predict_fn(x), surrogate)
    return surrogate


def test_true_ellipsoid_rho_skips_when_prediction_unavailable():
    tr = _build_true_tr(num_dim=3, rng_seed=7)
    surrogate = _make_surrogate(lambda _x: (_ for _ in ()).throw(RuntimeError("no pred")))
    x0 = np.array([0.2, 0.2, 0.2], dtype=float)
    x1 = np.array([0.3, 0.2, 0.2], dtype=float)
    surrogate._maybe_update_true_ellipsoid_rho(tr_state=tr, x_center=x0, y_obs=np.array([1.0]), incumbent_idx=0)
    with pytest.raises(RuntimeError, match="no pred"):
        surrogate._maybe_update_true_ellipsoid_rho(tr_state=tr, x_center=x1, y_obs=np.array([2.0]), incumbent_idx=0)
    assert tr._pending_rho is None


def test_true_ellipsoid_rho_sets_pending_with_valid_prediction():
    tr = _build_true_tr(num_dim=3, rng_seed=8)
    surrogate = _make_surrogate(lambda _x: _Posterior(mu=np.array([0.0, 1.0])))
    x0 = np.array([0.2, 0.2, 0.2], dtype=float)
    x1 = np.array([0.3, 0.2, 0.2], dtype=float)
    surrogate._maybe_update_true_ellipsoid_rho(tr_state=tr, x_center=x0, y_obs=np.array([1.0]), incumbent_idx=0)
    surrogate._maybe_update_true_ellipsoid_rho(tr_state=tr, x_center=x1, y_obs=np.array([2.0]), incumbent_idx=0)
    assert tr._pending_rho is not None


def test_update_trust_region_uses_capability_api_metric_path():
    surrogate = object.__new__(GeometryENNSurrogate)
    surrogate._enn = object()
    surrogate._params = object()

    called = {"observe": 0, "rho": 0}

    def _local(self, _x, *, params=None, exclude_nearest=True):
        _ = params, exclude_nearest
        return np.zeros((2, 3), dtype=float), np.array([0.5, 0.5], dtype=float)

    def _rho(self, **kwargs):
        _ = kwargs
        called["rho"] += 1

    surrogate.local_geometry = types.MethodType(_local, surrogate)
    surrogate._maybe_update_true_ellipsoid_rho = types.MethodType(_rho, surrogate)

    class _TR:
        def needs_gradient_signal(self):
            return False

        def observe_local_geometry(self, *, delta_x, weights, delta_y=None):
            _ = delta_x, weights, delta_y
            called["observe"] += 1

        def observe_incumbent_transition(self, **kwargs):
            _ = kwargs

    surrogate.update_trust_region(
        _TR(),
        x_center=np.array([0.0, 0.0, 0.0], dtype=float),
        y_obs=np.array([1.0], dtype=float),
        incumbent_idx=0,
        rng=np.random.default_rng(0),
    )
    assert called["observe"] == 1
    assert called["rho"] == 1


def test_update_trust_region_uses_capability_api_gradient_path():
    surrogate = object.__new__(GeometryENNSurrogate)
    surrogate._params = object()
    surrogate._enn = types.SimpleNamespace(num_outputs=1)

    called = {"observe": 0}

    geom = types.SimpleNamespace(
        delta_x=np.zeros((2, 3), dtype=float),
        y_neighbors=np.array([[2.0], [3.0]], dtype=float),
        weights=np.array([0.5, 0.5], dtype=float),
    )

    def _gradient_mu(self, _x, *, params=None, exclude_nearest=True):
        _ = params, exclude_nearest
        return None

    def _local_values(self, _x, *, params=None, exclude_nearest=True):
        _ = params, exclude_nearest
        return geom

    surrogate.gradient_mu = types.MethodType(_gradient_mu, surrogate)
    surrogate.local_geometry_values = types.MethodType(_local_values, surrogate)

    class _TR:
        def needs_gradient_signal(self):
            return True

        def observe_local_geometry(self, *, delta_x, weights, delta_y=None):
            _ = delta_x, weights
            called["observe"] += 1
            assert delta_y is not None
            assert delta_y.shape == (2,)

    surrogate.update_trust_region(
        _TR(),
        x_center=np.array([0.0, 0.0, 0.0], dtype=float),
        y_obs=np.array([1.0], dtype=float),
        incumbent_idx=0,
        rng=np.random.default_rng(0),
    )
    assert called["observe"] == 1


def test_update_trust_region_uses_analytic_gradient_when_available():
    surrogate = object.__new__(GeometryENNSurrogate)
    surrogate._params = object()
    surrogate._enn = types.SimpleNamespace(num_outputs=1)

    called = {"observe": 0, "grad": None}

    def _gradient_mu(self, _x, *, params=None, exclude_nearest=True):
        _ = params, exclude_nearest
        return np.array([1.0, -0.5, 0.0], dtype=float)

    def _rho(self, **kwargs):
        _ = kwargs

    surrogate.gradient_mu = types.MethodType(_gradient_mu, surrogate)
    surrogate._maybe_update_true_ellipsoid_rho = types.MethodType(_rho, surrogate)

    class _TR:
        def needs_gradient_signal(self):
            return True

        def observe_local_geometry(self, *, delta_x=None, weights=None, delta_y=None, grad=None):
            called["observe"] += 1
            called["grad"] = grad

    surrogate.update_trust_region(
        _TR(),
        x_center=np.array([0.0, 0.0, 0.0], dtype=float),
        y_obs=np.array([1.0], dtype=float),
        incumbent_idx=0,
        rng=np.random.default_rng(0),
    )
    assert called["observe"] == 1
    assert called["grad"] is not None
    np.testing.assert_array_almost_equal(called["grad"], [1.0, -0.5, 0.0])


def test_gradient_mu_returns_finite_gradient():
    """Analytic gradient_mu returns finite gradient with correct shape."""
    rng = np.random.default_rng(42)
    num_dim, n_obs, k = 4, 20, 8
    x_obs = rng.uniform(0.1, 0.9, size=(n_obs, num_dim))
    y_obs = np.sum(x_obs**2, axis=1, keepdims=True)

    cfg = ENNSurrogateConfig(k=k, fit=ENNFitConfig(num_fit_samples=10, num_fit_candidates=15))
    surrogate = GeometryENNSurrogate(cfg)
    surrogate.fit(x_obs, y_obs, rng=rng)

    x = rng.uniform(0.2, 0.8, size=num_dim)
    grad = surrogate.gradient_mu(x, exclude_nearest=True)
    assert grad is not None
    assert grad.shape == (num_dim,)
    assert np.all(np.isfinite(grad))
