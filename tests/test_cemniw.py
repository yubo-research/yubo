import numpy as np
from scipy.stats import multivariate_normal

from sampling.cemniw import CEMNIW


def _test_cemniw(mu_0, diag_cov_0, known_mu):
    np.random.seed(17)

    mu_0 = np.asarray(mu_0)
    cov_0 = np.diag(diag_cov_0)

    def likelihood(samples):
        # a query from some external source; probably slow to make
        probs = []
        for s in samples:
            assert s.x.min() >= 0 and s.x.max() <= 1
            probs.append(multivariate_normal(mean=mu_0, cov=cov_0).pdf(s.x))
        return probs

    cemniw = CEMNIW(
        mu_0=mu_0 if known_mu else 0.1 + 0 * mu_0,
        scale_0=0.3**2,
        known_mu=known_mu,
        sobol_seed=np.random.randint(9999),
    )

    num_samples = 20
    num_iter = 20 + 10 * len(mu_0) + 10 * int(not known_mu)
    for _ in range(num_iter):
        samples = cemniw.ask(num_samples)
        num_samples = min(30, 3 * num_samples)
        cemniw.tell(likelihood(samples), samples)

    xs = np.stack([s.x for s in cemniw.ask(1000)])
    mu_est = xs.mean(axis=0)
    cov_est = np.cov(xs.T)

    assert np.abs(mu_est - mu_0).max() < 0.01
    assert np.abs(cov_est - cov_0).max() < 0.1 * cov_0.max()


def test_cemiw_1d():
    _test_cemniw([0.3], [0.05**2], False)


def test_cemiw_2d():
    _test_cemniw([0.3, 0.4], np.array([0.05, 0.13]) ** 2, False)


def test_cemniw_1d():
    _test_cemniw([0.3], [0.05**2], True)


def test_cemniw_2d():
    _test_cemniw([0.3, 0.4], np.array([0.05, 0.13]) ** 2, True)
