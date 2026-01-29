import torch


def test_default_targeter():
    from acq.turbo_yubo.ty_model_factory import default_targeter

    x_center = torch.tensor([0.0, 0.0])
    x_target = torch.tensor([1.0, 1.0])
    result = default_targeter(x_center, x_target)
    assert torch.allclose(result, x_target)


def test_turbo_yubo_noop_model_init():
    from acq.turbo_yubo.ty_model_factory import TurboYUBONOOPModel

    train_x = torch.rand(10, 3)
    train_y = torch.rand(10)
    model = TurboYUBONOOPModel(train_x, train_y)
    assert model is not None


def test_turbo_yubo_noop_model_posterior():
    from acq.turbo_yubo.ty_model_factory import TurboYUBONOOPModel

    train_x = torch.rand(10, 3)
    train_y = torch.rand(10)
    model = TurboYUBONOOPModel(train_x, train_y)
    X = torch.rand(5, 3)
    posterior = model.posterior(X)
    sample = posterior.sample(torch.Size([2]))
    assert sample.shape[0] == 5
