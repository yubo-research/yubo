"""Metamorphic and unit tests for incremental BE ENN."""

from __future__ import annotations

import numpy as np
from enn.turbo.config.enn_index_driver import ENNIndexDriver

from optimizer.uhd_be_enn import (
    IncrementalBEEnn,
    acquisition_from_incremental,
    be_enn_selection_ready,
    fit_enn_batch_raw_reference,
    fit_enn_batch_reference,
    parse_be_acquisition,
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


def test_incremental_hnsw_disk_add_obs():
    rng = np.random.default_rng(0)
    d = 5
    reg = IncrementalBEEnn(
        k=3,
        num_fit_candidates=1,
        num_fit_samples=10,
        index_driver="hnsw_disk",
        rng=rng,
    )
    reg.add_obs(rng.standard_normal(d), 1.5)
    assert reg.obs_count == 1
    assert reg.model is not None
    assert reg.params is not None


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
    assert len(reg.model) == 8


def test_incremental_raw_posterior_finite_vs_batch_reference():
    rng = np.random.default_rng(5)
    d = 4
    enn_k = 3
    zs, ys = _synthetic_stream(rng, 10, d)
    reg = IncrementalBEEnn(
        k=enn_k,
        num_fit_candidates=1,
        num_fit_samples=10,
        fit_interval=1,
        batch_fit_candidates=200,
        batch_fit_samples=200,
        rng=np.random.default_rng(0),
    )
    for z, y in zip(zs, ys, strict=True):
        reg.add_obs(z, y)
    model, params = fit_enn_batch_raw_reference(
        zs,
        ys,
        enn_k,
        num_fit_candidates=200,
        num_fit_samples=200,
    )
    x_cand = rng.standard_normal((3, d))
    ucb_inc = ucb_from_incremental(reg, x_cand)
    from enn.enn.enn_params import PosteriorFlags

    post = model.posterior(x_cand, params=params, flags=PosteriorFlags(observation_noise=False))
    ucb_ref = np.asarray(post.mu).reshape(-1) + np.asarray(post.se).reshape(-1)
    assert ucb_inc.shape == ucb_ref.shape
    assert np.all(np.isfinite(ucb_inc))
    assert np.all(np.isfinite(ucb_ref))


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


def test_acquisition_from_incremental_mu_vs_ucb():
    rng = np.random.default_rng(7)
    d = 5
    zs, ys = _synthetic_stream(rng, 15, d)
    reg = IncrementalBEEnn(k=3, rng=np.random.default_rng(0))
    for z, y in zip(zs, ys, strict=True):
        reg.add_obs(z, y)
    x_cand = rng.standard_normal((4, d))
    mu, se = reg.predict(x_cand)
    mu_scores = acquisition_from_incremental(reg, x_cand, acquisition="mu")
    ucb_scores = acquisition_from_incremental(reg, x_cand, acquisition="ucb")
    assert np.allclose(mu_scores, mu)
    assert np.allclose(ucb_scores, mu + se)


def test_parse_be_acquisition_values():
    assert parse_be_acquisition("ucb") == "ucb"
    assert parse_be_acquisition("UCB") == "ucb"
    assert parse_be_acquisition(" mu ") == "mu"


def test_parse_be_acquisition_invalid():
    import pytest

    with pytest.raises(ValueError, match="Unknown be_acquisition"):
        parse_be_acquisition("bogus")


def test_batch_normalized_reference_and_ucb_helper():
    rng = np.random.default_rng(11)
    d = 3
    zs, ys = _synthetic_stream(rng, 8, d)
    model, params, y_mean, y_std = fit_enn_batch_reference(zs, ys, enn_k=3)
    x_cand = rng.standard_normal((2, d))
    ucb = ucb_from_batch_posterior(model, params, x_cand, y_mean, y_std)
    assert ucb.shape == (2,)
    assert np.all(np.isfinite(ucb))


def test_be_enn_selection_ready_respects_warmup_only():
    assert be_enn_selection_ready(obs_count=9, warmup=10, enn_k=15, has_params=True) is False
    assert be_enn_selection_ready(obs_count=10, warmup=10, enn_k=15, has_params=True) is True


def test_effective_k_caps_to_obs_count():
    reg = IncrementalBEEnn(k=15, rng=np.random.default_rng(0))
    reg.add_obs(np.zeros(3), 1.0)
    assert reg._effective_k() == 1
    for i in range(2, 12):
        reg.add_obs(np.ones(3) * i, float(i))
    assert reg._effective_k() == min(15, reg.obs_count - 1)


def test_fitter_y_std_matches_model_after_adds():
    import pytest

    rng = np.random.default_rng(9)
    d = 3
    reg = IncrementalBEEnn(k=3, rng=rng)
    ys = []
    for i in range(6):
        y = float(100 * i + rng.standard_normal())
        ys.append(y)
        reg.add_obs(rng.standard_normal(d), y)
    model_y = np.array([reg.model.train_rows_at([j])[1][0, 0] for j in range(reg.obs_count)])
    fitter_std = float(reg._fitter.y_std()[0])
    assert fitter_std == pytest.approx(float(model_y.std()), rel=1e-12)


def test_create_empty_factory():
    reg = IncrementalBEEnn.create_empty(4, k=2)
    assert reg.obs_count == 0
    assert reg.model is not None
