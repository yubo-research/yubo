import pytest
import torch

from uhd.tm_mnist import TMMNIST


def test_mnist_has_name_and_seed_type_assertion():
    assert getattr(TMMNIST, "name", None) == "tm_mnist"
    with pytest.raises(AssertionError):
        TMMNIST(seed="x")  # type: ignore[arg-type]


def test_mnist_deterministic_weights_with_seed():
    m1 = TMMNIST(seed=42)
    m2 = TMMNIST(seed=42)
    s1 = m1.state_dict()
    s2 = m2.state_dict()
    for k in s1:
        assert torch.allclose(s1[k], s2[k])


def test_mnist_forward_shape_cpu():
    m = TMMNIST(seed=0)
    m.eval()
    x = torch.randn(4, 1, 28, 28)
    y = m(x)
    assert y.shape == (4, 10)
    assert y.dtype == x.dtype


def test_mnist_device_migration_and_forward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TMMNIST(seed=7).to(device)
    m.eval()
    x = torch.randn(8, 1, 28, 28, device=device)
    y = m(x)
    assert y.shape == (8, 10)
    assert y.device.type == device
    for p in m.parameters():
        assert p.device.type == device
