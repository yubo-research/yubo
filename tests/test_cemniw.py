import numpy as np


def test_cemniw_init():
    from sampling.cemniw import CEMNIW

    mu_0 = np.array([0.5, 0.5])
    cem = CEMNIW(mu_0, scale_0=1.0)
    assert cem.num_dim == 2


def test_cemniw_estimate_mu_cov():
    from sampling.cemniw import CEMNIW

    mu_0 = np.array([0.5, 0.5])
    cem = CEMNIW(mu_0, scale_0=1.0)
    mu, cov = cem.estimate_mu_cov()
    assert mu.shape == (2,)
    assert cov.shape == (2,)


def test_cemniw_ask_init():
    from sampling.cemniw import CEMNIW

    np.random.seed(42)
    mu_0 = np.array([0.5, 0.5])
    cem = CEMNIW(mu_0, scale_0=1.0, sobol_seed=42)
    samples = cem.ask(num_samples=8)
    assert len(samples) == 8
