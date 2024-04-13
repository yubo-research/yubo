def test_appx_normal():
    import torch

    from acq.fit_gp import fit_gp_XY
    from sampling.appx_normal import AppxNormal

    torch.manual_seed(17)

    X = torch.rand(size=torch.Size([3, 2]))
    mu = torch.tensor([0.3, 0.3])
    Y = -((X - mu) ** 2).sum(dim=1)
    Y = Y + 0.25 * torch.randn(size=Y.shape)
    Y = Y[:, None]

    model = fit_gp_XY(X, Y)
    appx_normal = AppxNormal(
        model,
        mu,
        num_X_samples=64,
        num_Y_samples=128,
    )

    assert appx_normal._mk_p_star(torch.rand(size=torch.Size([10, 2]))).shape == (10,)
    X = appx_normal._resample_x(torch.tensor([10, 1]), torch.tensor([0.1, 10]))

    assert abs(X[:, 0].mean() - 10) < 0.1 / 3.1
    assert abs(X[:, 1].mean() - 1) < 10 / 3.1

    for _ in range(5):
        sigma = 0.2 * torch.rand(size=torch.Size([2]))
        print(mu, sigma, appx_normal.evalutate(sigma))


def test_appx_normal_func():
    import torch

    from acq.acq_util import find_max
    from acq.fit_gp import fit_gp_XY
    from sampling.appx_normal import appx_normal

    torch.manual_seed(16)

    X = torch.rand(size=torch.Size([10, 2]), dtype=torch.double)
    Y = -((X - 0.3) ** 2).sum(dim=1)
    # Y = Y + 0.3 * torch.randn(size=Y.shape)
    Y = Y[:, None]

    model = fit_gp_XY(X, Y)
    print()
    print("X_MAX:", find_max(model))
    seed = int(999999 * torch.rand(size=(1,)).item())
    for use_gradients in [True, False]:
        print(
            appx_normal(
                model,
                num_X_samples=64,
                num_Y_samples=128,
                use_gradients=use_gradients,
                seed=seed,
            )
        )
