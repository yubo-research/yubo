"""Tests for LABCAT-style PC rotation (optimizer.pc_rotation)."""

import numpy as np

from optimizer.pc_rotation import (
    LABCAT_CITATION,
    PCRotationResult,
    compute_labcat_weighted_pca,
)


def test_labcat_citation_present():
    assert "LABCAT" in LABCAT_CITATION
    assert "principal" in LABCAT_CITATION.lower() or "Principal" in LABCAT_CITATION


def test_compute_full_rotation():
    rng = np.random.default_rng(42)
    d, n = 8, 40
    x_center = np.zeros(d)
    x_obs = rng.normal(0, 0.1, size=(n, d))
    y_obs = rng.uniform(0, 1, size=n)
    result = compute_labcat_weighted_pca(x_center, x_obs, y_obs, maximize=True, mode="full")
    assert result.has_rotation
    assert result.basis.shape == (d, min(n, d))
    assert result.singular_values.shape == (min(n, d),)
    assert np.allclose(result.basis.T @ result.basis, np.eye(result.basis.shape[1]), atol=1e-10)


def test_compute_low_rank_rotation():
    rng = np.random.default_rng(43)
    d, n, k = 20, 50, 5
    x_center = np.zeros(d)
    x_obs = rng.normal(0, 0.05, size=(n, d))
    y_obs = rng.uniform(0, 1, size=n)
    result = compute_labcat_weighted_pca(x_center, x_obs, y_obs, maximize=True, mode="low_rank", rank=k)
    assert result.has_rotation
    assert result.basis.shape == (d, k)
    assert result.singular_values.shape == (k,)


def test_to_rotated_from_rotated_roundtrip():
    rng = np.random.default_rng(44)
    d, n = 6, 25
    x_center = rng.uniform(0.3, 0.7, size=d)
    x_obs = x_center + rng.normal(0, 0.05, size=(n, d))
    y_obs = rng.uniform(0, 1, size=n)
    result = compute_labcat_weighted_pca(x_center, x_obs, y_obs, maximize=True, mode="full")
    x_test = rng.uniform(0, 1, size=(10, d))
    x_rot = result.to_rotated(x_test)
    x_back = result.from_rotated(x_rot)
    np.testing.assert_allclose(x_test, x_back, atol=1e-10)


def test_insufficient_observations_returns_no_rotation():
    x_center = np.zeros(5)
    x_obs = np.random.randn(1, 5)  # only 1 obs
    y_obs = np.array([0.5])
    result = compute_labcat_weighted_pca(x_center, x_obs, y_obs, mode="full", min_obs=2)
    assert not result.has_rotation


def test_pc_rotation_result_direct_construction():
    center = np.zeros(3)
    basis = np.eye(3)
    singular_values = np.array([1.0, 0.5, 0.25])
    result = PCRotationResult(center=center, basis=basis, singular_values=singular_values, has_rotation=True)
    assert result.has_rotation
    assert result.center.shape == (3,)
    x = np.array([1.0, 2.0, 3.0])
    x_rot = result.to_rotated(x)
    np.testing.assert_allclose(x_rot, x)
