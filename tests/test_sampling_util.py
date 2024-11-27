def test_intersect_with_box():
    import numpy as np

    from sampling.sampling_util import intersect_with_box

    # Example usage
    x0 = np.array([0.2, 0.5])
    x1 = np.array([10, 1.5])
    intersection = intersect_with_box(x0, x1)
    print()
    print("I:", x0)
    print("O:", x1)
    print(intersection)


def test_var_of_var():
    import torch

    from sampling.sampling_util import var_of_var

    torch.manual_seed(17)
    n = 100
    for _ in range(5):
        w = torch.rand(size=(n,))
        w = w / w.sum()
        X = torch.randn(size=(n,))

        vv_w = var_of_var(w=w, X=X)

        w_0 = torch.ones(size=(n,))
        w_0 = w_0 / w_0.sum()
        vv_0 = var_of_var(w=w_0, X=X)

        if vv_w > vv_0:
            break
    else:
        assert False


def _test_draw_bounded_normal_samples(num_dim, qmc):
    import numpy as np
    import torch

    from sampling.sampling_util import draw_bounded_normal_samples

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

    from sampling.sampling_util import draw_bounded_normal_samples

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

    from sampling.sampling_util import draw_varied_bounded_normal_samples

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
