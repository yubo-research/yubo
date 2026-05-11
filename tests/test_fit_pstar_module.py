import numpy as np
import pytest


@pytest.fixture
def fit_pstar_fp():
    from sampling.fit_pstar import FitPStar

    mu = np.array([0.5, 0.5])
    cov_aspect = np.array([1.0, 1.0])
    return FitPStar(mu, cov_aspect, sigma_0=0.3)


def test_fit_pstar_init(fit_pstar_fp):
    assert np.isclose(fit_pstar_fp.sigma(), 0.3)


def test_fit_pstar_scale2(fit_pstar_fp):
    assert np.isclose(fit_pstar_fp.scale2(), 0.09)


def test_fit_pstar_cov(fit_pstar_fp):
    cov = fit_pstar_fp.cov()
    assert cov.shape == (2,)


def test_fit_pstar_mu(fit_pstar_fp):
    assert np.allclose(fit_pstar_fp.mu(), np.array([0.5, 0.5]))


def test_fit_pstar_estimate_scale2(fit_pstar_fp):
    mu = np.array([0.5, 0.5])
    x = np.random.rand(10, 2)
    scale2 = fit_pstar_fp.estimate_scale2(mu, x)
    assert scale2 > 0


def test_fit_pstar_ask(fit_pstar_fp):
    np.random.seed(42)
    x, p = fit_pstar_fp.ask(10)
    assert len(x) == 10
