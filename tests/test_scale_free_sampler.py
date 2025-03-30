def test_scale_free_sampler():
    import numpy as np

    from sampling.scale_free_sampler import scale_free_sampler

    for _ in range(10):
        X = np.random.uniform(size=(1, 4))
        X_prime = scale_free_sampler(X)
        dx = X_prime - X
        assert np.abs(dx).max() <= 1
        if (np.abs(dx) > 1e-6).sum() < 10:
            return
    assert False


def test_scale_free_sampler_multi():
    import numpy as np

    from sampling.scale_free_sampler import scale_free_sampler

    for b_raasp in [True, False]:
        X = np.random.uniform(size=(10, 5))
        x = scale_free_sampler(X, b_raasp=b_raasp)
        assert x.shape == (10, 5)
