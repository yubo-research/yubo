import torch


def test_standardizer_basic():
    from acq.standardizer import Standardizer

    Y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    s = Standardizer(Y)
    Y_std = s(Y)
    assert Y_std.mean().abs() < 1e-6
    assert (Y_std.std() - 1.0).abs() < 0.1


def test_standardizer_undo():
    from acq.standardizer import Standardizer

    Y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    s = Standardizer(Y)
    Y_std = s(Y)
    Y_back = s.undo(Y_std)
    assert torch.allclose(Y, Y_back, atol=1e-6)


def test_standardizer_2d():
    from acq.standardizer import Standardizer

    Y = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    s = Standardizer(Y)
    Y_std = s(Y)
    Y_back = s.undo(Y_std)
    assert torch.allclose(Y, Y_back, atol=1e-6)


def test_standardizer_constant():
    from acq.standardizer import Standardizer

    Y = torch.tensor([5.0, 5.0, 5.0])
    s = Standardizer(Y)
    Y_std = s(Y)
    assert torch.all(torch.isfinite(Y_std))
