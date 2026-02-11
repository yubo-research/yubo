"""Adversarial and stress tests for sparse JL transform.

Probes edge cases, dimensionality limits, speed scaling, and memory usage.
"""

import time
import tracemalloc

import torch
from torch import nn

from sampling.sparse_jl_t import block_sparse_jl_transform_module, block_sparse_jl_transform_t

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


def test_d_equals_1():
    x = torch.randn(100)
    y = block_sparse_jl_transform_t(x, d=1, s=1, seed=0)
    assert y.shape == (1,)
    assert torch.all(torch.isfinite(y))


def test_s_equals_d():
    x = torch.randn(50)
    y = block_sparse_jl_transform_t(x, d=8, s=8, seed=0)
    assert y.shape == (8,)
    assert torch.all(torch.isfinite(y))


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


def test_large_D_100k():
    D = 100_000
    d = 128
    s = 4
    x = torch.randn(D)
    y = block_sparse_jl_transform_t(x, d=d, s=s, seed=0)
    assert y.shape == (d,)
    assert torch.all(torch.isfinite(y))


def test_large_D_1M():
    D = 1_000_000
    d = 256
    s = 4
    x = torch.randn(D)
    y = block_sparse_jl_transform_t(x, d=d, s=s, seed=0)
    assert y.shape == (d,)
    assert torch.all(torch.isfinite(y))


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


# ---------------------------------------------------------------------------
# Speed scaling: verify O(D*s) time (not worse)
# ---------------------------------------------------------------------------


def test_speed_scaling_linear_in_D():
    """Time should scale roughly linearly with D."""
    d = 128
    s = 4
    seed = 0
    sizes = [50_000, 200_000, 800_000]
    times = []
    for D in sizes:
        x = torch.randn(D)
        # warmup
        block_sparse_jl_transform_t(x, d=d, s=s, seed=seed)
        t0 = time.perf_counter()
        for _ in range(3):
            block_sparse_jl_transform_t(x, d=d, s=s, seed=seed)
        t1 = time.perf_counter()
        times.append((t1 - t0) / 3)
    # Ratio of times should be roughly ratio of sizes
    # 200k/50k = 4x, 800k/200k = 4x
    # Allow generous 8x factor for overhead
    ratio_1 = times[1] / max(times[0], 1e-9)
    ratio_2 = times[2] / max(times[1], 1e-9)
    print(f"D=50k: {times[0] * 1000:.2f}ms, D=200k: {times[1] * 1000:.2f}ms, D=800k: {times[2] * 1000:.2f}ms")
    print(f"ratio 200k/50k: {ratio_1:.1f}x, ratio 800k/200k: {ratio_2:.1f}x")
    assert ratio_1 < 8, f"Time ratio 200k/50k = {ratio_1:.1f}x, expected < 8x"
    assert ratio_2 < 8, f"Time ratio 800k/200k = {ratio_2:.1f}x, expected < 8x"


def test_speed_scaling_linear_in_s():
    """Time should scale roughly linearly with s."""
    D = 200_000
    d = 256
    seed = 0
    x = torch.randn(D)
    s_values = [1, 2, 4, 8]
    times = []
    for s in s_values:
        block_sparse_jl_transform_t(x, d=d, s=s, seed=seed)
        t0 = time.perf_counter()
        for _ in range(3):
            block_sparse_jl_transform_t(x, d=d, s=s, seed=seed)
        t1 = time.perf_counter()
        times.append((t1 - t0) / 3)
    print(f"s=1: {times[0] * 1000:.2f}ms, s=2: {times[1] * 1000:.2f}ms, s=4: {times[2] * 1000:.2f}ms, s=8: {times[3] * 1000:.2f}ms")
    ratio = times[3] / max(times[0], 1e-9)
    print(f"ratio s=8/s=1: {ratio:.1f}x")
    # Fisher-Yates sorting adds O(s^2) per coordinate, so ratio > 8 is expected
    assert ratio < 30, f"s=8 vs s=1 ratio = {ratio:.1f}x, expected < 30x"


def test_speed_module_vs_tensor():
    """Module path should not be much slower than tensor path."""
    model = nn.Linear(500, 400, bias=True)
    total_d = sum(p.numel() for p in model.parameters())
    d, s, seed = 128, 4, 0
    params_flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    # Warmup
    block_sparse_jl_transform_t(params_flat, d=d, s=s, seed=seed)
    block_sparse_jl_transform_module(model, d=d, s=s, seed=seed)

    t0 = time.perf_counter()
    for _ in range(5):
        block_sparse_jl_transform_t(params_flat, d=d, s=s, seed=seed)
    t_tensor = (time.perf_counter() - t0) / 5

    t0 = time.perf_counter()
    for _ in range(5):
        block_sparse_jl_transform_module(model, d=d, s=s, seed=seed)
    t_module = (time.perf_counter() - t0) / 5

    ratio = t_module / max(t_tensor, 1e-9)
    print(f"D={total_d}, tensor: {t_tensor * 1000:.2f}ms, module: {t_module * 1000:.2f}ms, ratio: {ratio:.2f}x")
    assert ratio < 3, f"Module path {ratio:.1f}x slower than tensor, expected < 3x"


# ---------------------------------------------------------------------------
# Memory: verify O(d + chunk) not O(D)
# ---------------------------------------------------------------------------


def test_memory_does_not_scale_with_D():
    """Peak memory from transform should be bounded, not proportional to D."""
    d = 128
    s = 4
    seed = 0

    def measure_transform_memory(D):
        x = torch.randn(D)
        tracemalloc.start()
        _ = block_sparse_jl_transform_t(x, d=d, s=s, seed=seed)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    mem_100k = measure_transform_memory(100_000)
    mem_1m = measure_transform_memory(1_000_000)
    ratio = mem_1m / max(mem_100k, 1)
    print(f"mem 100k: {mem_100k / 1024:.0f}KB, mem 1M: {mem_1m / 1024:.0f}KB, ratio: {ratio:.1f}x")
    # If memory were O(D), ratio would be ~10x. We allow up to 5x for overhead.
    assert ratio < 5, f"Memory ratio 1M/100k = {ratio:.1f}x, suggests O(D) memory"


def test_memory_module_bounded():
    """Module transform memory should be bounded by chunk size, not total D."""
    d = 128
    s = 4
    seed = 0
    model = nn.Sequential(nn.Linear(500, 400), nn.Linear(400, 200))
    total_d = sum(p.numel() for p in model.parameters())

    tracemalloc.start()
    _ = block_sparse_jl_transform_module(model, d=d, s=s, seed=seed)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"D={total_d}, peak memory: {peak / 1024:.0f}KB")
    # Should be well under total_d * 4 bytes (float32)
    assert peak < total_d * 4, f"Peak {peak} >= D*4={total_d * 4}, suggests full copy"


# ---------------------------------------------------------------------------
# Hash quality: row distribution should be approximately uniform
# ---------------------------------------------------------------------------


def test_row_distribution_uniform():
    """For many random coordinates, rows should be near-uniform across [0, d)."""
    D = 10_000
    d = 64
    s = 4
    seed = 42
    # Use basis vectors and count which rows are hit
    row_counts = torch.zeros(d, dtype=torch.long)
    for j in range(D):
        x_j = torch.zeros(D)
        x_j[j] = 1.0
        y = block_sparse_jl_transform_t(x_j, d=d, s=s, seed=seed)
        row_counts += (y.abs() > 1e-12).long()
    # Expected count per row: D * s / d = 10000 * 4 / 64 = 625
    expected = D * s / d
    min_count = row_counts.min().item()
    max_count = row_counts.max().item()
    print(f"expected={expected:.0f}, min={min_count}, max={max_count}")
    # Allow 40% deviation
    assert min_count > expected * 0.6, f"Row {row_counts.argmin()} has only {min_count} hits"
    assert max_count < expected * 1.4, f"Row {row_counts.argmax()} has {max_count} hits"


# ---------------------------------------------------------------------------
# Sign distribution: +1 and -1 should be roughly balanced
# ---------------------------------------------------------------------------


def test_sign_distribution_balanced():
    D = 5_000
    d = 32
    s = 4
    seed = 99
    pos_count = 0
    neg_count = 0
    for j in range(D):
        x = torch.zeros(D)
        x[j] = 1.0
        y = block_sparse_jl_transform_t(x, d=d, s=s, seed=seed)
        nz = y[y.abs() > 1e-12]
        pos_count += (nz > 0).sum().item()
        neg_count += (nz < 0).sum().item()
    total = pos_count + neg_count
    frac_pos = pos_count / total
    print(f"positive fraction: {frac_pos:.3f} (expected ~0.5)")
    assert 0.45 < frac_pos < 0.55, f"Sign bias: {frac_pos:.3f}"


# ---------------------------------------------------------------------------
# Adversarial: linearity under extreme conditions
# ---------------------------------------------------------------------------


def test_linearity_large_D():
    D = 50_000
    d = 64
    s = 4
    seed = 42
    gen = torch.Generator().manual_seed(999)
    x = torch.randn(D, generator=gen)
    z = torch.randn(D, generator=gen)
    y_sum = block_sparse_jl_transform_t(x + z, d=d, s=s, seed=seed)
    y_x = block_sparse_jl_transform_t(x, d=d, s=s, seed=seed)
    y_z = block_sparse_jl_transform_t(z, d=d, s=s, seed=seed)
    assert torch.allclose(y_sum, y_x + y_z, atol=1e-4)


def test_linearity_scalar_mult():
    D = 1000
    d = 32
    s = 4
    seed = 7
    x = torch.randn(D)
    alpha = 3.7
    y_x = block_sparse_jl_transform_t(x, d=d, s=s, seed=seed)
    y_ax = block_sparse_jl_transform_t(alpha * x, d=d, s=s, seed=seed)
    # float32 rounding: alpha*(x*sign/sqrt(s)) != (alpha*x)*sign/sqrt(s)
    assert torch.allclose(y_ax, alpha * y_x, atol=1e-5)


# ---------------------------------------------------------------------------
# Adversarial: chunk boundary correctness
# ---------------------------------------------------------------------------


def test_chunk_boundary_correctness():
    """Verify results are identical regardless of internal chunking.

    We do this by checking module (multi-param) vs single tensor.
    The module path processes each param layer separately, so the chunk
    boundaries differ, but the global-index hashing ensures parity.
    """
    # Build a model where one layer is exactly _CHUNK_SIZE big
    from sampling.sparse_jl_t import _CHUNK_SIZE

    # Use a smaller test: verify D just above and below chunk size
    for D in [_CHUNK_SIZE - 1, _CHUNK_SIZE, _CHUNK_SIZE + 1]:
        x = torch.randn(D)
        y = block_sparse_jl_transform_t(x, d=64, s=4, seed=42)
        assert y.shape == (64,)
        assert torch.all(torch.isfinite(y))
        # Verify linearity still holds at chunk boundary
        z = torch.randn(D)
        y_sum = block_sparse_jl_transform_t(x + z, d=64, s=4, seed=42)
        y_x = block_sparse_jl_transform_t(x, d=64, s=4, seed=42)
        y_z = block_sparse_jl_transform_t(z, d=64, s=4, seed=42)
        # For D~1M, float32 scatter_add accumulates ~D*s/d terms per bin.
        # Accumulation-order rounding causes ~0.005 absolute error at these scales.
        assert torch.allclose(y_sum, y_x + y_z, rtol=1e-4, atol=0.01), f"Linearity broken at D={D}"


# ---------------------------------------------------------------------------
# Adversarial: embedding dimension larger than input
# ---------------------------------------------------------------------------


def test_d_much_larger_than_D():
    D = 5
    d = 1000
    s = 4
    x = torch.randn(D)
    y = block_sparse_jl_transform_t(x, d=d, s=s, seed=0)
    assert y.shape == (d,)
    nnz = (y.abs() > 1e-12).sum().item()
    # At most D * s nonzeros
    assert nnz <= D * s


# ---------------------------------------------------------------------------
# Print timing summary
# ---------------------------------------------------------------------------


def test_timing_summary_prints():
    """Print a timing table for various D values."""
    d = 128
    s = 4
    seed = 0
    print("\n--- Timing Summary ---")
    print(f"{'D':>12}  {'time_ms':>10}  {'throughput_M/s':>14}")
    for D in [10_000, 50_000, 200_000, 1_000_000]:
        x = torch.randn(D)
        # warmup
        block_sparse_jl_transform_t(x, d=d, s=s, seed=seed)
        t0 = time.perf_counter()
        reps = 3
        for _ in range(reps):
            block_sparse_jl_transform_t(x, d=d, s=s, seed=seed)
        elapsed = (time.perf_counter() - t0) / reps
        throughput = D / elapsed / 1e6
        print(f"{D:>12,}  {elapsed * 1000:>10.2f}  {throughput:>14.1f}")


def test_timing_module_summary_prints():
    """Print timing for module-based transform at various scales."""
    d = 128
    s = 4
    seed = 0
    print("\n--- Module Timing Summary ---")
    print(f"{'D':>12}  {'time_ms':>10}")
    configs = [(100, 50), (500, 200), (1000, 500)]
    for in_f, out_f in configs:
        model = nn.Sequential(nn.Linear(in_f, out_f), nn.Linear(out_f, out_f // 2))
        total = sum(p.numel() for p in model.parameters())
        block_sparse_jl_transform_module(model, d=d, s=s, seed=seed)
        t0 = time.perf_counter()
        for _ in range(3):
            block_sparse_jl_transform_module(model, d=d, s=s, seed=seed)
        elapsed = (time.perf_counter() - t0) / 3
        print(f"{total:>12,}  {elapsed * 1000:>10.2f}")
