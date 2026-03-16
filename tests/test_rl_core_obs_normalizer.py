"""Tests for SharedObsNormalizer."""

import numpy as np

from rl.core.obs_normalizer import SharedObsNormalizer


def test_shared_obs_normalizer_init():
    n = SharedObsNormalizer(obs_dim=3)
    assert n._mean.shape == (3,)
    assert n._var.shape == (3,)
    assert np.allclose(n._mean, 0.0)
    assert np.allclose(n._var, 1.0)


def test_shared_obs_normalizer_update_and_normalize():
    n = SharedObsNormalizer(obs_dim=2)
    obs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    n.update(obs)
    out = n.normalize(obs)
    assert out.shape == obs.shape
    assert out.dtype == np.float32
    # After update, normalize should produce zero-mean unit-var (approx) for the batch
    mean_out = out.mean(axis=0)
    assert np.allclose(mean_out, 0.0, atol=1e-5)
    std_out = out.std(axis=0)
    assert np.allclose(std_out, 1.0, atol=1e-5)


def test_shared_obs_normalizer_update_1d():
    n = SharedObsNormalizer(obs_dim=2)
    obs = np.array([1.0, 2.0], dtype=np.float64)
    n.update(obs)
    out = n.normalize(obs)
    assert out.shape == (2,)
    assert out.dtype == np.float32


def test_shared_obs_normalizer_clip():
    n = SharedObsNormalizer(obs_dim=2)
    n.update(np.array([[0.0, 0.0]], dtype=np.float64))
    # Large values should be clipped
    obs = np.array([[100.0, -100.0]], dtype=np.float32)
    out = n.normalize(obs, clip=10.0)
    assert np.all(out >= -10.0)
    assert np.all(out <= 10.0)
