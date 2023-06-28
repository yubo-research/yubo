import numpy as np
from scipy.stats import multivariate_normal

from sampling.cem_scale import CEMScale


def test_cem_scale():
    for num_dim in [1, 3, 10]:
        _test_cem_scale(num_dim)


def _test_cem_scale(num_dim):
    np.random.seed(17)

    sigma = 0.0314
    mu = np.asarray([0.5] * num_dim)
    unit_cov_diag = np.random.uniform(size=(num_dim,))
    adet = np.abs(np.prod(unit_cov_diag))
    unit_cov_diag = unit_cov_diag / (adet ** (1 / num_dim))

    adet = np.abs(np.prod(unit_cov_diag))
    assert (adet - 1) < 1e-6

    cov = (sigma**2) * unit_cov_diag

    def likelihood(samples):
        # a query from some external source; probably slow to make
        probs = []
        for s in samples:
            assert s.x.min() >= 0 and s.x.max() <= 1
            probs.append(multivariate_normal(mean=mu, cov=cov).pdf(s.x))
        return probs

    cem = CEMScale(
        mu=mu,
        unit_cov_diag=unit_cov_diag,
        sigma_0=3 * sigma,
        sobol_seed=np.random.randint(9999),
        alpha=0.2,
    )

    num_samples = 30
    num_iter = 100
    for _ in range(num_iter):
        samples = cem.ask(num_samples)
        assert len(samples) == num_samples, len(samples)
        cem.tell(likelihood(samples), samples)

    assert (cem.estimate_sigma() - 0.0314) < 0.01
