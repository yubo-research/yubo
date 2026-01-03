import math

import pytest
import torch

from uhd.tm_ackley import TMAckley


def test_ackley_deterministic_indices_with_seed():
    a1 = TMAckley(10, 3, seed=1234)
    a2 = TMAckley(10, 3, seed=1234)
    assert torch.equal(a1.active_idx, a2.active_idx)
    assert torch.allclose(a1.parameters_.data, a2.parameters_.data)


def test_ackley_forward_cpu():
    num_dim = 7
    num_active = 3
    a = TMAckley(num_dim, num_active, seed=0)
    a.parameters_.data = torch.arange(1.0, num_dim + 1)
    p = a.parameters_.index_select(0, a.active_idx)
    x_0 = a.x_0_active
    denom_neg = (x_0 - a.lb).clamp_min(1e-12)
    denom_pos = (a.ub - x_0).clamp_min(1e-12)
    y = torch.where(p < x_0, (p - x_0) / denom_neg, (p - x_0) / denom_pos)
    x = 32.768 * y
    d = x.numel()
    term1 = -20.0 * torch.exp(-0.2 * torch.sqrt(torch.sum(x * x) / d))
    term2 = -torch.exp(torch.sum(torch.cos(2.0 * math.pi * x)) / d)
    expected = -(term1 + term2 + 20.0 + math.e)
    out = a()
    assert torch.allclose(out, expected)


@pytest.mark.parametrize("num_dim,num_active", [(5, 0), (0, 1)])
def test_ackley_invalid_args(num_dim, num_active):
    with pytest.raises(AssertionError):
        TMAckley(num_dim, num_active, seed=0)


def test_ackley_device_migration_cpu_cuda_consistency():
    num_dim = 12
    num_active = 5
    a = TMAckley(num_dim, num_active, seed=42)
    a.parameters_.data = torch.randn(num_dim)
    p = a.parameters_.index_select(0, a.active_idx)
    x_0 = a.x_0_active
    denom_neg = (x_0 - a.lb).clamp_min(1e-12)
    denom_pos = (a.ub - x_0).clamp_min(1e-12)
    y = torch.where(p < x_0, (p - x_0) / denom_neg, (p - x_0) / denom_pos)
    x = 32.768 * y
    d = x.numel()
    term1 = -20.0 * torch.exp(-0.2 * torch.sqrt(torch.sum(x * x) / d))
    term2 = -torch.exp(torch.sum(torch.cos(2.0 * math.pi * x)) / d)
    expected = -(term1 + term2 + 20.0 + math.e)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = a.to(device)
    out = a()
    assert torch.allclose(out.detach().cpu(), expected.detach().cpu())
    assert a.parameters_.device == a.active_idx.device
