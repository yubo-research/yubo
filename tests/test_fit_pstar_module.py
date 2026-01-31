import numpy as np


def test_fit_pstar_init():
    from sampling.fit_pstar import FitPStar

    mu = np.array([0.5, 0.5])
    cov_aspect = np.array([1.0, 1.0])
    fp = FitPStar(mu, cov_aspect, sigma_0=0.3)
    assert np.isclose(fp.sigma(), 0.3)


def test_fit_pstar_scale2():
    from sampling.fit_pstar import FitPStar

    mu = np.array([0.5, 0.5])
    cov_aspect = np.array([1.0, 1.0])
    fp = FitPStar(mu, cov_aspect, sigma_0=0.3)
    assert np.isclose(fp.scale2(), 0.09)


def test_fit_pstar_cov():
    from sampling.fit_pstar import FitPStar

    mu = np.array([0.5, 0.5])
    cov_aspect = np.array([1.0, 1.0])
    fp = FitPStar(mu, cov_aspect, sigma_0=0.3)
    cov = fp.cov()
    assert cov.shape == (2,)


def test_fit_pstar_mu():
    from sampling.fit_pstar import FitPStar

    mu = np.array([0.5, 0.5])
    cov_aspect = np.array([1.0, 1.0])
    fp = FitPStar(mu, cov_aspect, sigma_0=0.3)
    assert np.allclose(fp.mu(), mu)


def test_fit_pstar_estimate_scale2():
    from sampling.fit_pstar import FitPStar

    mu = np.array([0.5, 0.5])
    cov_aspect = np.array([1.0, 1.0])
    fp = FitPStar(mu, cov_aspect, sigma_0=0.3)
    x = np.random.rand(10, 2)
    scale2 = fp.estimate_scale2(mu, x)
    assert scale2 > 0


def test_fit_pstar_ask():
    from sampling.fit_pstar import FitPStar

    np.random.seed(42)
    mu = np.array([0.5, 0.5])
    cov_aspect = np.array([1.0, 1.0])
    fp = FitPStar(mu, cov_aspect, sigma_0=0.3)
    x, p = fp.ask(10)
    assert len(x) == 10
