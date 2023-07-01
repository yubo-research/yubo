def test_mk_normal_samples():
    import numpy as np

    from sampling.util import mk_normal_samples

    mu = np.array([0.2, 0.8])
    cov = np.array([0.003, 0.01])
    mu_covs = [
        (mu, cov),
    ]
    num_samples = 100

    samples = mk_normal_samples(mu_covs, num_samples, qmc=False)
    x = np.array([s.x for s in samples])
    assert np.abs(x.mean(axis=0) - mu).max() < 0.05
    assert np.abs(x.var(axis=0) - cov).max() < 0.05

    samples = mk_normal_samples(mu_covs, num_samples, qmc=True)
    x = np.array([s.x.numpy() for s in samples])
    assert np.abs(x.mean(axis=0) - mu).max() < 0.05
    assert np.abs(x.var(axis=0) - cov).max() < 0.05


def test_mk_normal_samples_high_dim():
    import numpy as np

    from sampling.util import mk_normal_samples

    num_dim = 100
    mu = np.random.uniform(size=(num_dim,))
    cov = 0.01 * np.ones(shape=(num_dim,))
    cov[0] = 0.003

    mu_covs = [
        (mu, cov),
    ]

    num_samples = 64

    samples = mk_normal_samples(mu_covs, num_samples, qmc=False)
    x = np.array([s.x for s in samples])
    assert np.abs(x.mean(axis=0) - mu).max() < 0.05
    assert np.abs(x.var(axis=0) - cov).max() < 0.05
