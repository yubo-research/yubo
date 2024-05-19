def test_appx_trunc_normal():
    import torch

    from sampling.appx_trunc_normal import _AppxTruncNormal
    from tests.test_util import gp_parabola

    torch.manual_seed(17)

    model, mu = gp_parabola()
    for use_soft_max in [True, False]:
        appx_normal = _AppxTruncNormal(
            model,
            num_X_samples=64,
            num_Y_samples=128,
            use_soft_max=True,
            theta=0.1,
        )

        assert appx_normal._mk_p_star(torch.rand(size=torch.Size([10, 2]))).shape == (10,)
        X, p = appx_normal._sample_trunc_normal(torch.tensor([0.9, 0.25]), torch.tensor([0.1, 2]))

        assert abs(X[:, 0].mean() - 0.9) < 0.1 / 3.1
        assert abs(X[:, 1].mean() - 0.15) < 2 / 3.1
        assert torch.all(p >= 0) and torch.all(p <= 1)
        for _ in range(5):
            sigma = 0.2 * torch.rand(size=torch.Size([2]))
            print(mu, sigma, appx_normal.evaluate(mu, sigma))


def test_appx_normal_func():
    import time

    from acq.acq_util import find_max
    from sampling.appx_trunc_normal import appx_trunc_normal
    from tests.test_util import gp_parabola

    # torch.manual_seed(16)

    model = gp_parabola(num_samples=3)[0]

    print()
    x_max = find_max(model)
    print("X_MAX:", x_max)
    # seed = int(999999 * torch.rand(size=(1,)).item())
    for use_gradients in [True, False]:
        t_0 = time.time()
        an = appx_trunc_normal(
            model,
            num_X_samples=256,
            num_Y_samples=1024,
            num_tries=30,
            use_gradients=use_gradients,
        )
        print("TIME:", time.time() - t_0)
        print("MS:", an.mu, an.sigma)
        # assert np.abs(an.mu.numpy() - x_max.numpy()).max() < 1e-4
        X = an.sample(num_X_samples=32)
        assert ((an.calc_importance_weights(X) - 1) ** 2).mean() < 30
