def test_appx_normal():
    import torch

    from sampling.appx_normal import _AppxNormal
    from tests.test_util import gp_parabola

    torch.manual_seed(17)

    model, mu = gp_parabola()
    for use_soft_max in [True, False]:
        appx_normal = _AppxNormal(
            model,
            mu,
            num_X_samples=64,
            num_Y_samples=128,
            use_soft_max=True,
        )

        assert appx_normal._mk_p_star(torch.rand(size=torch.Size([10, 2]))).shape == (10,)
        X = appx_normal._sample_normal(torch.tensor([10, 1]), torch.tensor([0.1, 10]))

        assert abs(X[:, 0].mean() - 10) < 0.1 / 3.1
        assert abs(X[:, 1].mean() - 1) < 10 / 3.1

        for _ in range(5):
            sigma = 0.2 * torch.rand(size=torch.Size([2]))
            print(mu, sigma, appx_normal.evalutate(sigma))


def test_appx_normal_func():
    import numpy as np
    import torch

    from acq.acq_util import find_max
    from sampling.appx_normal import appx_normal
    from tests.test_util import gp_parabola

    torch.manual_seed(16)

    model = gp_parabola(num_samples=10)[0]

    print()
    x_max = find_max(model)
    print("X_MAX:", x_max)
    seed = int(999999 * torch.rand(size=(1,)).item())
    for use_gradients in [True, False]:
        an = appx_normal(
            model,
            num_X_samples=32,
            num_Y_samples=256,
            use_gradients=use_gradients,
            seed=seed,
        )
        print("MS:", an.mu, an.sigma)
        assert np.abs(an.mu.numpy() - x_max.numpy()).max() < 1e-4
        X = an.sample(num_X_samples=32)
        assert ((an.calc_importance_weights(model, X, num_Y_samples=1024) - 1) ** 2).mean() < 30
