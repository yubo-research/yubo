import pytest
import torch

from uhd.tm_sphere import TMSphere


def test_sphere_deterministic_indices_with_seed():
    s1 = TMSphere(10, 3, seed=1234)
    s2 = TMSphere(10, 3, seed=1234)
    assert torch.equal(s1.active_idx, s2.active_idx)
    assert torch.allclose(s1.x_0, s2.x_0)
    assert torch.allclose(s1.parameters_.data, s2.parameters_.data)


def test_sphere_forward_sum_of_squares_cpu():
    num_dim = 7
    num_active = 3
    s = TMSphere(num_dim, num_active, seed=0)
    s.parameters_.data = torch.arange(1.0, num_dim + 1)
    x = s.parameters_.index_select(0, s.active_idx)
    expected = -((x - s.x_0) * (x - s.x_0)).sum()
    out = s()
    assert torch.allclose(out, expected)


@pytest.mark.parametrize("num_dim,num_active", [(5, 0), (0, 1)])
def test_sphere_invalid_args(num_dim, num_active):
    with pytest.raises(AssertionError):
        TMSphere(num_dim, num_active, seed=0)


def test_sphere_device_migration_cpu_cuda_consistency():
    num_dim = 12
    num_active = 5
    s = TMSphere(num_dim, num_active, seed=42)
    s.parameters_.data = torch.randn(num_dim)
    x = s.parameters_.index_select(0, s.active_idx)
    expected = -((x - s.x_0) * (x - s.x_0)).sum()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = s.to(device)
    out = s()
    assert torch.allclose(out.detach().cpu(), expected.detach().cpu())
    assert s.parameters_.device == s.active_idx.device
