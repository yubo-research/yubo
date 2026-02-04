from __future__ import annotations

import numpy as np
import pytest
from enn.turbo.config.candidate_rv import CandidateRV
from enn.turbo.config.enn_surrogate_config import ENNFitConfig, ENNSurrogateConfig
from scipy.stats import qmc

from optimizer.ellipsoid_trust_region import (
    EllipsoidTRConfig,
    EllipsoidTrustRegionBase,
    ENNEllipsoidTrustRegion,
)
from optimizer.enn_surrogate_ext import GeometryENNSurrogate, LocalGeometryValues


def _build_tr(num_dim: int, *, geometry: str = "enn_ellipsoid", sampler=None, rank=None):
    rng = np.random.default_rng(0)
    cfg = EllipsoidTRConfig(
        geometry=geometry,
        ellipsoid_sampler=sampler,
        ellipsoid_rank=rank,
    )
    assert cfg.length_init == cfg.length.length_init
    assert cfg.length_min == cfg.length.length_min
    assert cfg.length_max == cfg.length.length_max
    tr = cfg.build(num_dim=num_dim, rng=rng, candidate_rv=CandidateRV.SOBOL)
    assert isinstance(tr, EllipsoidTrustRegionBase)
    assert isinstance(tr, ENNEllipsoidTrustRegion)
    return tr


def test_ellipsoid_candidates_are_sparse_in_identity_geometry():
    num_dim = 80
    tr = _build_tr(num_dim)
    rng = np.random.default_rng(0)
    sobol = qmc.Sobol(d=num_dim, scramble=True, seed=0)
    x_center = np.full(num_dim, 0.5, dtype=float)
    candidates = tr.generate_candidates(
        x_center,
        None,
        256,
        rng=rng,
        candidate_rv=CandidateRV.SOBOL,
        sobol_engine=sobol,
        num_pert=20,
    )
    assert candidates.shape == (256, num_dim)
    assert np.all(candidates >= 0.0) and np.all(candidates <= 1.0)
    diff = np.abs(candidates - x_center.reshape(1, -1)) > 1e-12
    num_changed = diff.sum(axis=1)
    assert float(np.mean(num_changed)) > 5.0
    assert float(np.mean(num_changed)) < 35.0


def test_ellipsoid_set_geometry_updates_factor():
    tr = _build_tr(2)
    assert not tr.has_enn_geometry
    delta_x = np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 0.1], [0.0, -0.1]])
    weights = np.ones(delta_x.shape[0], dtype=float)
    tr.set_geometry(delta_x=delta_x, weights=weights)
    assert tr.has_enn_geometry
    cov = tr._cov_factor @ tr._cov_factor.T
    assert not np.allclose(cov, np.eye(2, dtype=float))

    tr.restart(rng=np.random.default_rng(0))
    assert not tr.has_geometry


def test_ellipsoid_gradient_geometry_prefers_large_slopes():
    tr = _build_tr(3, geometry="enn_grad_ellipsoid")
    assert not tr.has_enn_geometry
    delta_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    delta_y = np.array([10.0, -10.0, 0.5, -0.5], dtype=float)
    weights = np.ones(delta_x.shape[0], dtype=float)
    tr.set_gradient_geometry(delta_x=delta_x, delta_y=delta_y, weights=weights)
    assert tr.has_enn_geometry
    sqrt_alpha, basis, _sqrt_vals = tr._low_rank
    assert sqrt_alpha > 0.0
    assert int(np.argmax(np.abs(basis[:, 0]))) == 0


def test_ellipsoid_low_rank_initialization():
    tr = _build_tr(3, sampler="low_rank")
    sqrt_alpha, basis, sqrt_vals = tr._low_rank
    assert sqrt_alpha == 1.0
    assert basis.shape == (3, 0)
    assert sqrt_vals.shape == (0,)
    assert not tr.has_enn_geometry


def test_ellipsoid_low_rank_respects_rank_cap():
    tr = _build_tr(5, sampler="low_rank", rank=2)
    delta_x = np.random.default_rng(42).normal(0, 0.1, size=(10, 5))
    weights = np.ones(10, dtype=float)
    tr.set_geometry(delta_x=delta_x, weights=weights)
    _sqrt_alpha, basis, sqrt_vals = tr._low_rank
    assert basis.shape[1] <= 2
    assert sqrt_vals.shape[0] == basis.shape[1]


def test_surrogate_local_geometry_values_returns_offsets_and_weights():
    x_train = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    y_train = np.array([0.0, 1.0, 1.0, 2.0], dtype=float).reshape(-1, 1)
    cfg = ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=5, num_fit_candidates=10))
    surrogate = GeometryENNSurrogate(cfg)
    surrogate.fit(x_train, y_train, rng=np.random.default_rng(0))
    geom = surrogate.local_geometry_values(x_train[0])
    assert isinstance(geom, LocalGeometryValues)
    delta_x, y_neighbors, weights = geom.delta_x, geom.y_neighbors, geom.weights
    assert delta_x.shape == (3, 2)
    assert y_neighbors.shape == (3, 1)
    assert weights.shape == (3,)
    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0)

    delta_x2, weights2 = surrogate.local_geometry(x_train[0])
    assert delta_x2.shape == (3, 2)
    assert weights2.shape == (3,)
    assert np.all(weights2 >= 0.0)
    assert np.isclose(weights2.sum(), 1.0)


def test_surrogate_grad_geometry_requires_scalar_y():
    x_train = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    y_train = np.array([0.0, 1.0, 1.0, 2.0], dtype=float).reshape(-1, 1)
    cfg = ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=5, num_fit_candidates=10))
    surrogate = GeometryENNSurrogate(cfg)
    surrogate.fit(x_train, y_train, rng=np.random.default_rng(0))
    tr_cfg = EllipsoidTRConfig(geometry="enn_grad_ellipsoid")
    tr = tr_cfg.build(num_dim=2, rng=np.random.default_rng(0), candidate_rv=CandidateRV.SOBOL)
    y_obs = np.concatenate([y_train, y_train], axis=1)
    with pytest.raises(ValueError, match="scalar"):
        surrogate.update_trust_region(tr, x_train[0], y_obs, incumbent_idx=0, rng=np.random.default_rng(0))
