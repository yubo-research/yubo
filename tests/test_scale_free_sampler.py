def test_scale_free_sampler():
    import torch

    from sampling.scale_free_sampler import scale_free_sampler

    for _ in range(10):
        X = torch.rand(size=(1, 4))
        X_prime = scale_free_sampler(X)
        dx = X_prime - X
        assert torch.abs(dx).max() <= 1
        if (torch.abs(dx) > 1e-6).sum() < 10:
            return
    assert False


def test_scale_free_sampler_multi():
    import torch

    from sampling.scale_free_sampler import scale_free_sampler

    X = torch.rand(size=(10, 5))
    x = scale_free_sampler(X)
    assert x.shape == (10, 5)
