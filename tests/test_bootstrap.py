def test_bootstrap():
    # See https://medium.com/@v.lahtinen/weighted-bayesian-bootstrap-f90b4550ee2c
    import torch

    from sampling.bootstrap import boot_means

    num_dim = 3
    x = torch.rand(size=(10, num_dim))

    b_x = boot_means(x, num_boot=1000)
    m = b_x.mean(axis=0)
    se = b_x.std(axis=0)

    assert torch.abs(m - x.mean(axis=0)).max() < se.max()
