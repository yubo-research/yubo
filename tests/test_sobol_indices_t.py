import pytest
import torch

from sampling.sobol_indices import calculate_sobol_indices_np


def maybe_cuda():
    return "cuda" if torch.cuda.is_available() else "cpu"


def gen_data(n, d, device, dtype):
    g = torch.Generator(device="cpu").manual_seed(1234)
    x = torch.rand((n, d), generator=g, dtype=dtype, device="cpu").to(device)
    # y depends nonlinearly on x to get nontrivial indices
    w = torch.linspace(0.5, 1.5, d, dtype=dtype, device=device)
    y = (x.pow(2) * w).sum(dim=1) + 0.1 * torch.randn(n, dtype=dtype, device=device)
    return x, y


def to_numpy(x_t, y_t):
    return x_t.detach().cpu().numpy(), y_t.detach().cpu().numpy()


def ref_np(x_t, y_t):
    x_np, y_np = to_numpy(x_t, y_t)
    return calculate_sobol_indices_np(x_np, y_np)


def import_torch_impl():
    from sampling.sobol_indices_t import calculate_sobol_indices_t

    return calculate_sobol_indices_t


@pytest.mark.parametrize("n", [9, 15, 60])
@pytest.mark.parametrize("d", [1, 3, 7])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_parity_with_numpy(n, d, dtype):
    device = "cpu"
    x_t, y_t = gen_data(n, d, device, dtype)
    ref = ref_np(x_t, y_t)
    calculate_sobol_indices_t = import_torch_impl()
    out_t = calculate_sobol_indices_t(x_t, y_t)
    assert isinstance(out_t, torch.Tensor)
    assert out_t.shape == (d,)
    torch.testing.assert_close(out_t.cpu(), torch.from_numpy(ref), atol=1e-5, rtol=1e-5)


def test_accepts_y_column_vector_and_1d():
    n, d = 40, 4
    x_t, y_1d = gen_data(n, d, "cpu", torch.float32)
    y_2d = y_1d[:, None]
    calculate_sobol_indices_t = import_torch_impl()
    s1 = calculate_sobol_indices_t(x_t, y_1d)
    s2 = calculate_sobol_indices_t(x_t, y_2d)
    torch.testing.assert_close(s1, s2, atol=0, rtol=0)


def test_small_n_returns_ones():
    n, d = 5, 6
    x_t, y_t = gen_data(n, d, "cpu", torch.float32)
    calculate_sobol_indices_t = import_torch_impl()
    out = calculate_sobol_indices_t(x_t, y_t)
    assert out.shape == (d,)
    torch.testing.assert_close(out, torch.ones_like(out), atol=0, rtol=0)


def test_zero_variance_y_returns_ones():
    n, d = 30, 5
    x_t, _ = gen_data(n, d, "cpu", torch.float64)
    y_t = torch.zeros(n, dtype=torch.float64)
    calculate_sobol_indices_t = import_torch_impl()
    out = calculate_sobol_indices_t(x_t, y_t)
    torch.testing.assert_close(out, torch.ones_like(out), atol=0, rtol=0)


def test_constant_feature_yields_zero_index():
    n, d = 50, 4
    x_t, y_t = gen_data(n, d - 1, "cpu", torch.float32)
    const_col = torch.zeros((n, 1), dtype=torch.float32)
    x_t = torch.cat([x_t, const_col], dim=1)
    calculate_sobol_indices_t = import_torch_impl()
    out = calculate_sobol_indices_t(x_t, y_t)
    assert out.shape == (d,)
    assert torch.isfinite(out).all()
    assert out[-1].abs() < 1e-6


def test_row_permutation_invariance():
    n, d = 45, 6
    x_t, y_t = gen_data(n, d, "cpu", torch.float32)
    perm = torch.randperm(n)
    x_p = x_t[perm]
    y_p = y_t[perm]
    calculate_sobol_indices_t = import_torch_impl()
    s1 = calculate_sobol_indices_t(x_t, y_t)
    s2 = calculate_sobol_indices_t(x_p, y_p)
    torch.testing.assert_close(s1, s2, atol=2e-7, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_matches_cpu_within_tolerance():
    n, d = 60, 5
    x_cpu, y_cpu = gen_data(n, d, "cpu", torch.float32)
    x_gpu = x_cpu.cuda()
    y_gpu = y_cpu.cuda()
    calculate_sobol_indices_t = import_torch_impl()
    s_cpu = calculate_sobol_indices_t(x_cpu, y_cpu)
    s_gpu = calculate_sobol_indices_t(x_gpu, y_gpu)
    torch.testing.assert_close(s_gpu.cpu(), s_cpu.cpu(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_output_dtype_and_device_follow_input(dtype):
    device = maybe_cuda() if dtype == torch.float32 else "cpu"
    x_t, y_t = gen_data(35, 4, device, dtype)
    calculate_sobol_indices_t = import_torch_impl()
    out = calculate_sobol_indices_t(x_t, y_t)
    assert out.device == x_t.device
    assert out.dtype == x_t.dtype
