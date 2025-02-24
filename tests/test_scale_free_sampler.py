def test_scale_free_sampler():
    import torch

    from sampling.scale_free_sampler import scale_free_sampler

    for _ in range(5):
        X = torch.rand(size=(1, 10))
        X_prime = scale_free_sampler(X)
        dx = X_prime - X
        assert torch.abs(dx).max() <= 1
        if (torch.abs(dx) > 1e-6).sum() < 10:
            return
    assert False
