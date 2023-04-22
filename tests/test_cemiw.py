import numpy as np
from scipy.stats import multivariate_normal

from sampling.cemiw import CEMIW


def _test_cemiw(mu_0, diag_cov_0):
    mu_0 = np.asarray(mu_0)
    cov_0 = np.diag(diag_cov_0)

    def likelihood(samples):
        # a query from some external source; probably slow to make
        p = [multivariate_normal(mean=mu_0, cov=cov_0).pdf(s.x) for s in samples]
        return p

    cemiw = CEMIW(mu=mu_0, scale_0=0.3**2)  # mu_0 is known  # init scale_0 to any value

    np.random.seed(17)
    num_samples = 10
    for _ in range(10):
        samples = cemiw.ask(num_samples)
        num_samples = min(1000, 3 * num_samples)
        cemiw.tell(likelihood(samples), samples)

    xs = np.stack([s.x for s in cemiw.ask(1000)])
    mu_est = xs.mean(axis=0)  # recovered, really
    cov_est = np.cov(xs.T)

    assert np.abs(mu_est - mu_0).max() < 0.01
    assert np.abs(cov_est - cov_0).max() < 0.1 * cov_0.max()


def test_cemiw_1d():
    _test_cemiw([0.3], [0.05**2])


def test_cemiw_2d():
    _test_cemiw([0.3, 0.4], np.array([0.05, 0.13]) ** 2)
