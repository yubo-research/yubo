import torch


def test_gp_init():
    from gpytorch.constraints.constraints import Interval
    from gpytorch.likelihoods import GaussianLikelihood

    from acq.fit_gp_turbo import GP

    train_x = torch.tensor([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], dtype=torch.float64)
    train_y = torch.tensor([1.0, 2.0, 1.5], dtype=torch.float64)

    lengthscale_constraint = Interval(0.005, 2.0)
    outputscale_constraint = Interval(0.05, 20.0)
    likelihood = GaussianLikelihood()

    gp = GP(
        train_x,
        train_y,
        likelihood,
        lengthscale_constraint,
        outputscale_constraint,
        ard_dims=2,
    )
    assert gp is not None


def test_gp_forward():
    from gpytorch.constraints.constraints import Interval
    from gpytorch.likelihoods import GaussianLikelihood

    from acq.fit_gp_turbo import GP

    train_x = torch.tensor([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], dtype=torch.float64)
    train_y = torch.tensor([1.0, 2.0, 1.5], dtype=torch.float64)

    lengthscale_constraint = Interval(0.005, 2.0)
    outputscale_constraint = Interval(0.05, 20.0)
    likelihood = GaussianLikelihood().double()

    gp = GP(
        train_x,
        train_y,
        likelihood,
        lengthscale_constraint,
        outputscale_constraint,
        ard_dims=2,
    )
    gp.eval()

    test_x = torch.tensor([[0.3, 0.3]], dtype=torch.float64)
    output = gp(test_x)
    assert output.mean.shape == (1,)


def test_train_gp():
    from acq.fit_gp_turbo import train_gp

    train_x = torch.tensor([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], dtype=torch.float64)
    train_y = torch.tensor([1.0, 2.0, 1.5], dtype=torch.float64)

    gp = train_gp(train_x, train_y, use_ard=True, num_steps=10)
    assert gp is not None
