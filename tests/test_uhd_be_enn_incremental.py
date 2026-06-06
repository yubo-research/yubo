"""Metamorphic and unit tests for incremental BE ENN."""

from __future__ import annotations

import numpy as np
from enn.turbo.config.enn_index_driver import ENNIndexDriver

from optimizer.uhd_be_enn import (
    IncrementalBEEnn,
    fit_enn_batch_raw_reference,
    fit_enn_batch_reference,
    parse_be_enn_index_driver,
    ucb_from_batch_posterior,
    ucb_from_incremental,
)


def _synthetic_stream(rng: np.random.Generator, n: int, d: int):
    zs = []
    ys = []
    for _ in range(n):
        z = rng.standard_normal(d)
        y = float(np.sin(z[0]) + 0.1 * rng.standard_normal())
        zs.append(z)
        ys.append(y)
    return zs, ys


def test_incremental_add_obs_builds_model_and_params():
    rng = np.random.default_rng(0)
    d = 5
    reg = IncrementalBEEnn(k=3, num_fit_candidates=1, num_fit_samples=10, rng=rng)
    z, y = rng.standard_normal(d), 1.5
    reg.add_obs(z, y)
    assert reg.obs_count == 1
    assert reg.model is not None
    assert reg.params is not None


def test_incremental_predict_shape():
    rng = np.random.default_rng(1)
    d = 4
    reg = IncrementalBEEnn(k=3, rng=rng)
    zs, ys = _synthetic_stream(rng, 12, d)
    for z, y in zip(zs, ys, strict=True):
        reg.add_obs(z, y)
    x_cand = rng.standard_normal((7, d))
    mu, se = reg.predict(x_cand)
    assert mu.shape == (7,)
    assert se.shape == (7,)


def test_incremental_reuses_same_model_instance():
    rng = np.random.default_rng(42)
    d = 6
    reg = IncrementalBEEnn(k=3, rng=rng)
    model_ids = []
    for i in range(8):
        z = rng.standard_normal(d)
        reg.add_obs(z, float(i))
        model_ids.append(id(reg.model))
    assert len(set(model_ids)) == 1
    assert reg.obs_count == 8


def test_incremental_final_posterior_finite_on_candidates():
    rng = np.random.default_rng(7)
    d = 5
    enn_k = 3
    zs, ys = _synthetic_stream(rng, 15, d)
    reg = IncrementalBEEnn(k=enn_k, rng=np.random.default_rng(0))
    for z, y in zip(zs, ys, strict=True):
        reg.add_obs(z, y)
    x_cand = rng.standard_normal((4, d))
    ucb = ucb_from_incremental(reg, x_cand)
    assert ucb.shape == (4,)
    assert np.all(np.isfinite(ucb))


def test_batch_raw_reference_produces_valid_posterior():
    rng = np.random.default_rng(3)
    d = 4
    zs, ys = _synthetic_stream(rng, 10, d)
    model, params = fit_enn_batch_raw_reference(zs, ys, enn_k=3)
    x_cand = rng.standard_normal((3, d))
    from enn.enn.enn_params import PosteriorFlags

    post = model.posterior(x_cand, params=params, flags=PosteriorFlags(observation_noise=False))
    ucb = np.asarray(post.mu).reshape(-1) + np.asarray(post.se).reshape(-1)
    assert np.all(np.isfinite(ucb))


def test_parse_be_enn_index_driver_values():
    assert parse_be_enn_index_driver("flat") == ENNIndexDriver.FLAT
    assert parse_be_enn_index_driver("hnsw") == ENNIndexDriver.HNSW
    assert parse_be_enn_index_driver("hnsw_disk") == ENNIndexDriver.HNSW_DISK


def test_parse_be_enn_index_driver_invalid():
    import pytest

    with pytest.raises(ValueError, match="Unknown be_enn_index_driver"):
        parse_be_enn_index_driver("bogus")


def test_batch_normalized_reference_and_ucb_helper():
    rng = np.random.default_rng(11)
    d = 3
    zs, ys = _synthetic_stream(rng, 8, d)
    model, params, y_mean, y_std = fit_enn_batch_reference(zs, ys, enn_k=3)
    x_cand = rng.standard_normal((2, d))
    ucb = ucb_from_batch_posterior(model, params, x_cand, y_mean, y_std)
    assert ucb.shape == (2,)
    assert np.all(np.isfinite(ucb))


def test_create_empty_factory():
    reg = IncrementalBEEnn.create_empty(4, k=2)
    assert reg.obs_count == 0
    assert reg.model is not None
