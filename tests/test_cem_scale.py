import numpy as np


def test_cem_scale_init():
    from sampling.cem_scale import CEMScale

    mu = np.array([0.5, 0.5])
    cov_aspect = np.array([1.0, 1.0])
    cem = CEMScale(mu, cov_aspect, sigma_0=0.1)
    assert cem.sigma() == 0.1


def test_cem_scale_ask():
    from sampling.cem_scale import CEMScale

    np.random.seed(42)
    mu = np.array([0.5, 0.5])
    cov_aspect = np.array([1.0, 1.0])
    cem = CEMScale(mu, cov_aspect, sigma_0=0.1)
    x, p = cem.ask(10)
    assert x.shape == (10, 2)
    assert len(p) == 10


def test_cem_scale_tell():
    from sampling.cem_scale import CEMScale

    np.random.seed(42)
    mu = np.array([0.5, 0.5])
    cov_aspect = np.array([1.0, 1.0])
    cem = CEMScale(mu, cov_aspect, sigma_0=0.1)
    x, p = cem.ask(10)
    p_max = np.random.rand(10)
    cem.tell(x, p, p_max)
    assert cem.sigma() > 0
