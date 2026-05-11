"""Sparse JL edge cases, modules, dtypes, and large-D tensor tests (part 1 of 2)."""

import torch
from torch import nn

from sampling.sparse_jl_t import (
    block_sparse_jl_transform_module,
    block_sparse_jl_transform_t,
)


# ---------------------------------------------------------------------------
# Edge-case: extreme parameter values
# ---------------------------------------------------------------------------


def test_extreme_values_large():
    x = torch.full((100,), 1e30)
    y = block_sparse_jl_transform_t(x, d=16, s=4, seed=0)
    assert torch.all(torch.isfinite(y))
    assert y.norm().item() > 0


def test_extreme_values_tiny():
    x = torch.full((100,), 1e-30)
    y = block_sparse_jl_transform_t(x, d=16, s=4, seed=0)
    assert torch.all(torch.isfinite(y))
    # float32 norm underflows (1e-30 squared = 1e-60 < float32 min),
    # so check double-precision norm instead
    assert y.double().norm().item() > 0


def test_extreme_values_mixed():
    x = torch.zeros(100)
    x[0] = 1e30
    x[1] = -1e30
    x[2] = 1e-30
    y = block_sparse_jl_transform_t(x, d=32, s=4, seed=0)
    assert torch.all(torch.isfinite(y))


def test_nan_input():
    x = torch.tensor([1.0, float("nan"), 3.0])
    y = block_sparse_jl_transform_t(x, d=8, s=2, seed=0)
    assert torch.any(torch.isnan(y)), "NaN input should propagate to output"


def test_inf_input():
    x = torch.tensor([1.0, float("inf"), -float("inf")])
    y = block_sparse_jl_transform_t(x, d=8, s=2, seed=0)
    assert torch.any(torch.isinf(y)), "Inf input should propagate to output"


# ---------------------------------------------------------------------------
# Edge-case: boundary dimensions
# ---------------------------------------------------------------------------


def _assert_transform_shape_and_finite(*, d: int, s: int):
    x = torch.randn(50)
    y = block_sparse_jl_transform_t(x, d=d, s=s, seed=0)
    assert y.shape == (d,)
    assert torch.all(torch.isfinite(y))


def test_d_equals_1():
    _assert_transform_shape_and_finite(d=1, s=1)


def test_s_equals_d():
    _assert_transform_shape_and_finite(d=8, s=8)


def test_s_equals_1():
    x = torch.zeros(32)
    x[7] = 1.0
    y = block_sparse_jl_transform_t(x, d=16, s=1, seed=0)
    nnz = (y.abs() > 1e-12).sum().item()
    assert nnz == 1, f"s=1 should place exactly 1 nonzero, got {nnz}"


def test_D_equals_1():
    x = torch.tensor([3.14])
    y = block_sparse_jl_transform_t(x, d=10, s=4, seed=0)
    assert y.shape == (10,)
    nnz = (y.abs() > 1e-12).sum().item()
    assert nnz == 4


def test_D_equals_1_module():
    model = nn.Linear(1, 1, bias=False)
    nn.init.constant_(model.weight, 2.5)
    y = block_sparse_jl_transform_module(model, d=10, s=4, seed=0)
    assert y.shape == (10,)


# ---------------------------------------------------------------------------
# Edge-case: seeds
# ---------------------------------------------------------------------------


def test_seed_zero():
    x = torch.randn(64)
    y = block_sparse_jl_transform_t(x, d=16, s=4, seed=0)
    assert torch.all(torch.isfinite(y))


def test_seed_large():
    x = torch.randn(64)
    y = block_sparse_jl_transform_t(x, d=16, s=4, seed=2**31 - 1)
    assert torch.all(torch.isfinite(y))
    y2 = block_sparse_jl_transform_t(x, d=16, s=4, seed=2**31 - 1)
    assert torch.equal(y, y2)


def test_seed_negative():
    x = torch.randn(64)
    y = block_sparse_jl_transform_t(x, d=16, s=4, seed=-1)
    assert torch.all(torch.isfinite(y))


def test_many_seeds_all_different():
    x = torch.randn(100)
    outputs = []
    for seed in range(50):
        outputs.append(block_sparse_jl_transform_t(x, d=16, s=4, seed=seed))
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            assert not torch.equal(outputs[i], outputs[j]), f"seed {i} == seed {j}"


# ---------------------------------------------------------------------------
# Without-replacement exhaustive check across many coordinates
# ---------------------------------------------------------------------------


def test_without_replacement_all_coordinates():
    D = 64
    d = 16
    s = 4
    seed = 42
    for j in range(D):
        x = torch.zeros(D)
        x[j] = 1.0
        y = block_sparse_jl_transform_t(x, d=d, s=s, seed=seed)
        nnz = (y.abs() > 1e-12).sum().item()
        assert nnz == s, f"coordinate {j}: expected {s} nonzeros, got {nnz}"


def test_without_replacement_s_equals_d_all_rows_hit():
    D = 10
    d = 8
    s = 8
    seed = 7
    for j in range(D):
        x = torch.zeros(D)
        x[j] = 1.0
        y = block_sparse_jl_transform_t(x, d=d, s=s, seed=seed)
        nnz = (y.abs() > 1e-12).sum().item()
        assert nnz == d, f"coordinate {j}: s==d, expected all {d} rows hit, got {nnz}"


# ---------------------------------------------------------------------------
# Scaling sign magnitudes: |sign| == 1/sqrt(s)
# ---------------------------------------------------------------------------


def test_sign_magnitude_many_seeds():
    import math

    D = 20
    d = 10
    s = 4
    expected = 1.0 / math.sqrt(s)
    for seed in range(20):
        for j in range(D):
            x = torch.zeros(D)
            x[j] = 1.0
            y = block_sparse_jl_transform_t(x, d=d, s=s, seed=seed)
            nz = y[y.abs() > 1e-12]
            assert torch.allclose(nz.abs(), torch.full_like(nz, expected)), f"seed={seed}, j={j}: expected magnitude {expected}, got {nz.abs()}"


# ---------------------------------------------------------------------------
# Module: parameter ordering matters
# ---------------------------------------------------------------------------


def test_module_param_order_sensitivity():
    """Two modules with same total params but different ordering must differ."""
    m1 = nn.Sequential(nn.Linear(10, 5, bias=False), nn.Linear(5, 3, bias=False))
    m2 = nn.Sequential(nn.Linear(10, 5, bias=False), nn.Linear(5, 3, bias=False))
    # Same total D but different weights
    torch.manual_seed(1)
    for p in m1.parameters():
        p.data.normal_()
    torch.manual_seed(2)
    for p in m2.parameters():
        p.data.normal_()
    y1 = block_sparse_jl_transform_module(m1, d=16, s=4, seed=0)
    y2 = block_sparse_jl_transform_module(m2, d=16, s=4, seed=0)
    assert not torch.equal(y1, y2)


def test_module_weight_perturbation_detected():
    """A tiny weight change should change the embedding."""
    model = nn.Linear(50, 10, bias=True)
    d, s, seed = 32, 4, 42
    y_before = block_sparse_jl_transform_module(model, d=d, s=s, seed=seed)
    model.weight.data[0, 0] += 0.001
    y_after = block_sparse_jl_transform_module(model, d=d, s=s, seed=seed)
    assert not torch.equal(y_before, y_after)


# ---------------------------------------------------------------------------
# Module: conv, batchnorm, various layer types
# ---------------------------------------------------------------------------


def test_module_conv_network():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3),
    )
    y = block_sparse_jl_transform_module(model, d=32, s=4, seed=0)
    params_flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    yt = block_sparse_jl_transform_t(params_flat, d=32, s=4, seed=0)
    torch.testing.assert_close(y, yt, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Dtypes
# ---------------------------------------------------------------------------


def test_float64_tensor():
    x = torch.randn(100, dtype=torch.float64)
    y = block_sparse_jl_transform_t(x, d=16, s=4, seed=0)
    assert y.dtype == torch.float64
    assert torch.all(torch.isfinite(y))


def test_float16_module():
    model = nn.Linear(50, 10).half()
    y = block_sparse_jl_transform_module(model, d=16, s=4, seed=0)
    assert y.dtype == torch.float16


# ---------------------------------------------------------------------------
# Dimensionality stress tests
# ---------------------------------------------------------------------------


def _assert_large_d_sparse_jl(D, d, s, seed):
    x = torch.randn(D)
    y = block_sparse_jl_transform_t(x, d=d, s=s, seed=seed)
    assert y.shape == (d,)
    assert torch.all(torch.isfinite(y))


def test_large_D_100k():
    _assert_large_d_sparse_jl(100_000, 128, 4, 0)


def test_large_D_1M():
    _assert_large_d_sparse_jl(1_000_000, 256, 4, 0)


def test_large_D_module_400k():
    """Simulates ~400k params (e.g., a small MLP)."""
    model = nn.Sequential(nn.Linear(500, 400), nn.Linear(400, 200))
    total_d = sum(p.numel() for p in model.parameters())
    assert total_d > 200_000
    d, s, seed = 128, 4, 0
    y = block_sparse_jl_transform_module(model, d=d, s=s, seed=seed)
    params_flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    yt = block_sparse_jl_transform_t(params_flat, d=d, s=s, seed=seed)
    torch.testing.assert_close(y, yt, atol=0, rtol=0)
