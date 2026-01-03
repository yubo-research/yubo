import time

import pytest
import torch

from model.enn import EpistemicNearestNeighbors
from model.enn_t import ENNNormalT, EpistemicNearestNeighborsT


def build_numpy_reference(train_x_t, train_y_t, x_t, k, exclude_nearest):
    enn = EpistemicNearestNeighbors(k=k)
    enn.add(train_x_t.cpu().numpy(), train_y_t.cpu().numpy())
    mvn = enn.posterior(x_t.cpu().numpy(), k=k, exclude_nearest=exclude_nearest)
    return torch.from_numpy(mvn.mu), torch.from_numpy(mvn.se)


def generate_data(num_points, num_dim, num_metrics, device, dtype):
    g = torch.Generator(device=device).manual_seed(1337)
    train_x = torch.rand((num_points, num_dim), generator=g, device=device, dtype=dtype)
    train_y = torch.randn((num_points, num_metrics), generator=g, device=device, dtype=dtype)
    return train_x, train_y


def test_ennnormalt_sample_shapes():
    num_dim = 3
    mu = torch.randn(1, num_dim)
    se = torch.rand(1, num_dim) + 1e-3
    ennt = ENNNormalT(mu=mu, se=se)
    assert ennt.sample(num_samples=1).shape == (1, num_dim, 1)
    assert ennt.sample(num_samples=2).shape == (1, num_dim, 2)
    assert ennt.sample(num_samples=100).shape == (1, num_dim, 100)


def test_len_and_add_increments():
    k = 3
    num_points = 10
    num_dim = 5
    num_metrics = 2
    train_x, train_y = generate_data(num_points, num_dim, num_metrics, device="cpu", dtype=torch.float32)
    train_y_var = torch.zeros_like(train_y)
    model_t = EpistemicNearestNeighborsT(k=k)
    n0 = len(model_t)
    model_t.add(train_x[:0], train_y[:0], train_y_var[:0])
    assert len(model_t) == n0
    model_t.add(train_x[:1], train_y[:1], train_y_var[:1])
    assert len(model_t) == n0 + 1
    model_t.add(train_x[1:], train_y[1:], train_y_var[1:])
    assert len(model_t) == n0 + num_points


@pytest.mark.parametrize("exclude_nearest", [False, True])
@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_posterior_matches_numpy_reference(k, exclude_nearest):
    num_points = 30
    num_dim = 4
    num_metrics = 3
    dtype = torch.float32
    device = "cpu"
    train_x, train_y = generate_data(num_points, num_dim, num_metrics, device=device, dtype=dtype)
    train_y_var = torch.zeros_like(train_y)
    x = torch.rand((7, num_dim), device=device, dtype=dtype)
    model_t = EpistemicNearestNeighborsT(k=k)
    model_t.add(train_x, train_y, train_y_var)
    ref_mu, ref_se = build_numpy_reference(train_x, train_y, x, k=k, exclude_nearest=exclude_nearest)
    out = model_t.posterior(x, k=k, exclude_nearest=exclude_nearest)
    assert isinstance(out, ENNNormalT)
    torch.testing.assert_close(out.mu.cpu(), ref_mu, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out.se.cpu(), ref_se, atol=1e-5, rtol=1e-5)


def test_posterior_empty_training_returns_zeros_and_ones():
    num_dim = 3
    num_metrics = 2
    x = torch.randn(5, num_dim)
    model_t = EpistemicNearestNeighborsT(k=3)
    y = torch.randn(0, num_metrics)
    y_var = torch.zeros_like(y)
    model_t.add(x[:0], y, y_var)
    out = model_t.posterior(x, k=3)
    assert isinstance(out, ENNNormalT)
    assert out.mu.shape == (x.shape[0], num_metrics)
    assert out.se.shape == (x.shape[0], num_metrics)
    assert torch.allclose(out.mu, torch.zeros_like(out.mu), atol=0, rtol=0)
    assert torch.allclose(out.se, torch.ones_like(out.se), atol=0, rtol=0)


def test_posterior_single_observation_unit_se():
    num_dim = 2
    num_metrics = 1
    x = torch.randn(1, num_dim)
    y = torch.randn(1, num_metrics)
    y_var = torch.zeros_like(y)
    model_t = EpistemicNearestNeighborsT(k=3)
    model_t.add(x, y, y_var)
    out = model_t.posterior(x, k=3)
    assert isinstance(out, ENNNormalT)
    assert out.mu.shape == (1, num_metrics)
    assert out.se.shape == (1, num_metrics)
    torch.testing.assert_close(out.mu, y, atol=1e-6, rtol=0)
    torch.testing.assert_close(out.se, torch.ones_like(out.se), atol=1e-6, rtol=0)


def test_posterior_two_observations_reasonable():
    num_metrics = 1
    x_train = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    y_train = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    y_var_train = torch.zeros_like(y_train)
    x_query = torch.tensor([[0.5]], dtype=torch.float32)
    model_t = EpistemicNearestNeighborsT(k=3)
    model_t.add(x_train, y_train, y_var_train)
    out = model_t.posterior(x_query, k=3)
    assert isinstance(out, ENNNormalT)
    assert out.mu.shape == (1, num_metrics)
    assert out.se.shape == (1, num_metrics)
    assert 0.0 < out.mu.item() < 1.0
    assert out.se.item() > 0.0


def test_k_truncates_to_dataset_size():
    num_points = 4
    num_dim = 3
    num_metrics = 1
    train_x, train_y = generate_data(num_points, num_dim, num_metrics, device="cpu", dtype=torch.float32)
    train_y_var = torch.zeros_like(train_y)
    x = torch.randn(2, num_dim)
    model_t = EpistemicNearestNeighborsT(k=10)
    model_t.add(train_x, train_y, train_y_var)
    out = model_t.posterior(x, k=10)
    assert isinstance(out, ENNNormalT)
    ref_mu, ref_se = build_numpy_reference(train_x, train_y, x, k=num_points, exclude_nearest=False)
    torch.testing.assert_close(out.mu.cpu(), ref_mu, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out.se.cpu(), ref_se, atol=1e-5, rtol=1e-5)


def test_exact_match_k1_returns_neighbor_y_within_tolerance():
    k = 1
    train_x = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    train_y = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    train_y_var = torch.zeros_like(train_y)
    model_t = EpistemicNearestNeighborsT(k=k)
    model_t.add(train_x, train_y, train_y_var)
    out = model_t.posterior(train_x, k=k, exclude_nearest=False)
    assert isinstance(out, ENNNormalT)
    torch.testing.assert_close(out.mu, train_y, atol=1e-6, rtol=0)


def test_exclude_nearest_requires_at_least_two_points():
    num_dim = 2
    num_metrics = 1
    model_t = EpistemicNearestNeighborsT(k=2)
    train_x = torch.rand(1, num_dim)
    train_y = torch.rand(1, num_metrics)
    train_y_var = torch.zeros_like(train_y)
    model_t.add(train_x, train_y, train_y_var)
    with pytest.raises(AssertionError):
        _ = model_t.posterior(train_x, k=1, exclude_nearest=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_outputs_are_on_same_device_and_dtype_cpu(dtype):
    num_points = 20
    num_dim = 5
    num_metrics = 3
    device = "cpu"
    train_x, train_y = generate_data(num_points, num_dim, num_metrics, device=device, dtype=dtype)
    train_y_var = torch.zeros_like(train_y)
    x = torch.randn(6, num_dim, device=device, dtype=dtype)
    model_t = EpistemicNearestNeighborsT(k=3)
    model_t.add(train_x, train_y, train_y_var)
    out = model_t.posterior(x, k=3)
    assert isinstance(out, ENNNormalT)
    assert out.mu.device == x.device
    assert out.se.device == x.device
    assert out.mu.dtype == dtype
    assert out.se.dtype == dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_device_behavior_and_numerical_parity():
    num_points = 40
    num_dim = 6
    num_metrics = 2
    dtype = torch.float32
    cpu_train_x, cpu_train_y = generate_data(num_points, num_dim, num_metrics, device="cpu", dtype=dtype)
    x_cpu = torch.rand(8, num_dim, dtype=dtype)
    ref_mu, ref_se = build_numpy_reference(cpu_train_x, cpu_train_y, x_cpu, k=4, exclude_nearest=False)
    gpu_train_x = cpu_train_x.cuda()
    gpu_train_y = cpu_train_y.cuda()
    gpu_train_y_var = torch.zeros_like(gpu_train_y)
    x_gpu = x_cpu.cuda()
    model_t = EpistemicNearestNeighborsT(k=4)
    model_t.add(gpu_train_x, gpu_train_y, gpu_train_y_var)
    out = model_t.posterior(x_gpu, k=4, exclude_nearest=False)
    assert isinstance(out, ENNNormalT)
    assert out.mu.device.type == "cuda"
    assert out.se.device.type == "cuda"
    torch.testing.assert_close(out.mu.cpu(), ref_mu, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out.se.cpu(), ref_se, atol=1e-4, rtol=1e-4)


def test_epistemic_nearest_neighbors_t_timing_k10_n1000():
    g = torch.Generator(device="cpu").manual_seed(2025)
    n = 1000
    q = 200
    d = 8
    m = 1
    x_train = torch.rand((n, d), generator=g)
    y_train = torch.randn((n, m), generator=g)
    y_var_train = torch.zeros_like(y_train)
    x_query = torch.rand((q, d), generator=g)

    start = time.perf_counter()
    for _ in range(1000):
        model_t = EpistemicNearestNeighborsT(k=10)
        model_t.add(x_train, y_train, y_var_train)
        _ = model_t.posterior(x_query, k=10, exclude_nearest=False)
    elapsed = time.perf_counter() - start
    print(f"EPENN_T_TIMING_LOOP k=10 n={n} q={q} iters=100 dt={elapsed:.6f}s")
    assert elapsed > 0.0
