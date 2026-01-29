import torch


def test_standardize_torch_basic():
    from acq.fit_gp import standardize_torch

    Y = torch.tensor([[1.0], [2.0], [3.0]])
    Y_std = standardize_torch(Y)
    assert Y_std.shape == Y.shape
    assert torch.isclose(Y_std.mean(), torch.tensor(0.0), atol=1e-5)


def test_standardize_torch_empty():
    from acq.fit_gp import standardize_torch

    Y = torch.tensor([]).reshape(0, 1)
    Y_std = standardize_torch(Y)
    assert Y_std.shape == Y.shape


def test_standardize_torch_single():
    from acq.fit_gp import standardize_torch

    Y = torch.tensor([[5.0]])
    Y_std = standardize_torch(Y)
    assert Y_std.shape == Y.shape
    assert torch.all(Y_std == 0)


def test_untransform_posterior():
    from acq.fit_gp import _EmptyTransform

    transform = _EmptyTransform()
    mock_posterior = object()
    result = transform.untransform_posterior(mock_posterior)
    assert result is mock_posterior


def test_empty_transform_forward():
    from acq.fit_gp import _EmptyTransform

    transform = _EmptyTransform()
    Y = torch.tensor([[1.0], [2.0]])
    Y_out, Yvar_out = transform.forward(Y)
    assert torch.equal(Y_out, Y)
    assert Yvar_out is None


def test_empty_transform_untransform():
    from acq.fit_gp import _EmptyTransform

    transform = _EmptyTransform()
    Y = torch.tensor([[1.0], [2.0]])
    Y_out, Yvar_out = transform.untransform(Y)
    assert torch.equal(Y_out, Y)
    assert Yvar_out is None
