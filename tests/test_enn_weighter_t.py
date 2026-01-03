import pytest
import torch

from model.enn_weighter import ENNWeighter
from model.enn_weighter_t import ENNWeighterT


def gen_train(n, d, device, dtype):
    g = torch.Generator(device="cpu").manual_seed(4242)
    x = torch.rand((n, d), generator=g, dtype=dtype, device="cpu").to(device)
    y = torch.randn((n, 1), generator=g, dtype=dtype, device="cpu").to(device)
    return x, y


def as_numpy(x_t, y_t):
    return x_t.detach().cpu().numpy(), y_t.detach().cpu().numpy()


@pytest.mark.parametrize("weighting", ["sobol_indices", "sigma_x", "sobol_over_sigma"])
def test_weights_match_numpy_reference(weighting):
    n, d = 40, 6
    device, dtype = "cpu", torch.float32
    x_t, y_t = gen_train(n, d, device, dtype)
    x_np, y_np = as_numpy(x_t, y_t)
    ref = ENNWeighter(weighting=weighting, k=3)
    ref.add(x_np, y_np)
    w_ref = ref.weights
    dut = ENNWeighterT(weighting=weighting, k=3)
    dut.add(x_t, y_t)
    w_t = dut.weights
    assert torch.is_tensor(w_t)
    assert w_t.shape == (d,)
    torch.testing.assert_close(w_t.cpu(), torch.from_numpy(w_ref), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("weighting", ["sobol_indices", "sigma_x", "sobol_over_sigma"])
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("exclude_nearest", [False, True])
def test_posterior_matches_numpy_reference(weighting, k, exclude_nearest):
    n, d = 50, 5
    if exclude_nearest and n < 2:
        pytest.skip("requires at least two points")
    device, dtype = "cpu", torch.float32
    x_t, y_t = gen_train(n, d, device, dtype)
    x_np, y_np = as_numpy(x_t, y_t)
    ref = ENNWeighter(weighting=weighting, k=3)
    ref.add(x_np, y_np)
    dut = ENNWeighterT(weighting=weighting, k=3)
    dut.add(x_t, y_t)
    q = torch.rand((7, d), dtype=dtype, device=device)
    mvn_ref = ref.posterior(q.detach().cpu().numpy(), k=k, exclude_nearest=exclude_nearest)
    mvn_dut = dut.posterior(q, k=k, exclude_nearest=exclude_nearest)
    torch.testing.assert_close(mvn_dut.mu.cpu(), torch.from_numpy(mvn_ref.mu), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(mvn_dut.se.cpu(), torch.from_numpy(mvn_ref.se), atol=1e-5, rtol=1e-5)


def test_add_only_once():
    n, d = 10, 3
    x_t, y_t = gen_train(n, d, "cpu", torch.float32)
    dut = ENNWeighterT(weighting="sigma_x", k=2)
    dut.add(x_t, y_t)
    with pytest.raises(AssertionError):
        dut.add(x_t, y_t)


def test_small_n_sobol_indices_returns_uniform_weights():
    n, d = 8, 4
    x_t, y_t = gen_train(n, d, "cpu", torch.float32)
    dut = ENNWeighterT(weighting="sobol_indices", k=3)
    dut.add(x_t, y_t)
    w = dut.weights
    torch.testing.assert_close(w, torch.full((d,), 1.0 / d, dtype=w.dtype, device=w.device), atol=0, rtol=0)


def test_sigma_x_constant_feature_has_larger_weight_than_varying_feature():
    n, d = 40, 3
    g = torch.Generator(device="cpu").manual_seed(7)
    x = torch.rand((n, d - 1), generator=g)
    const = torch.zeros((n, 1))
    x = torch.cat([x, const], dim=1)
    y = torch.randn((n, 1), generator=g)
    dut = ENNWeighterT(weighting="sigma_x", k=3)
    dut.add(x, y)
    w = dut.weights
    assert w[-1] > w[0]


def test_unsupported_weighting_raises():
    with pytest.raises(AssertionError):
        _ = ENNWeighterT(weighting="curvature", k=3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_dtype_gpu_paths():
    n, d = 60, 5
    device = "cuda"
    dtype = torch.float32
    x_t, y_t = gen_train(n, d, device, dtype)
    dut = ENNWeighterT(weighting="sobol_over_sigma", k=3)
    dut.add(x_t, y_t)
    w = dut.weights
    assert w.device.type == "cuda"
    q = torch.rand((11, d), device=device, dtype=dtype)
    mvn = dut.posterior(q, k=4)
    assert mvn.mu.device.type == "cuda"
    assert mvn.se.device.type == "cuda"
