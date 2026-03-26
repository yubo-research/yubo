import torch

from rl.math_utils import atanh


def test_atanh_basic():
    x = torch.tensor([0.0, 0.5, -0.5])
    y = atanh(x)
    expected = torch.atanh(x)
    assert torch.allclose(y, expected, atol=1e-5)


def test_atanh_near_bounds():
    x = torch.tensor([0.999, -0.999])
    y = atanh(x)
    assert torch.isfinite(y).all()


def test_atanh_at_bounds_clamped():
    x = torch.tensor([1.0, -1.0])
    y = atanh(x)
    assert torch.isfinite(y).all()
