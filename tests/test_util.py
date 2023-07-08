def _test_draw_bounded_normal_samples(num_dim, qmc):
    import numpy as np
    import torch

    from sampling.util import draw_bounded_normal_samples

    np.random.seed(17)
    torch.manual_seed(17)

    mu = np.random.uniform(size=(num_dim,))
    cov = 0.003 * np.ones(shape=(num_dim,))
    cov[0] = 0.001

    num_samples = 1024

    x, p = draw_bounded_normal_samples(mu, cov, num_samples, qmc=qmc)

    assert np.abs(x.mean(axis=0) - mu).max() < 0.05
    assert np.abs(x.var(axis=0) - cov).max() < 0.05


def test_narrow():
    for qmc in [True, False]:
        for num_dim in [1, 3, 10, 30, 100]:
            _test_draw_bounded_normal_samples(num_dim, qmc)


def test_wide():
    import numpy as np

    from sampling.util import draw_bounded_normal_samples

    np.random.seed(17)

    num_dim = 100
    mu = np.random.uniform(size=(num_dim,))
    cov = 0.3 * np.ones(shape=(num_dim,))
    cov[0] = 0.1

    num_samples = 1024

    x, _ = draw_bounded_normal_samples(mu, cov, num_samples, qmc=False)

    assert x.min() >= 0 and x.max() <= 1


def _test_draw_varied_bounded_normal_samples(num_dim):
    import numpy as np
    import torch

    from sampling.util import draw_varied_bounded_normal_samples

    np.random.seed(17)
    torch.manual_seed(17)

    mu = np.random.uniform(size=(num_dim,))
    cov = 0.003 * np.ones(shape=(num_dim,))
    cov[0] = 0.001

    mus_covs = [
        (mu, cov),
        (mu, cov),
        (mu, cov),
    ]
    x, p = draw_varied_bounded_normal_samples(mus_covs)

    assert x.shape == (len(mus_covs), num_dim)
    assert len(p) == len(mus_covs)


def _xx_test_varied_1():
    _test_draw_varied_bounded_normal_samples(1)


def _xx_test_varied_n():
    for n in [2, 3, 10]:
        _test_draw_varied_bounded_normal_samples(n)
