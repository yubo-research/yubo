from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl.core import runtime as core_runtime


def test_obs_scaler_affine_and_dtype_guard():
    scaler = core_runtime.ObsScaler(
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.asarray([2.0, 4.0], dtype=np.float32),
    )
    out = scaler(torch.as_tensor([[3.0, 10.0]], dtype=torch.float32))
    assert np.allclose(out.detach().cpu().numpy(), np.asarray([[1.0, 2.0]], dtype=np.float32))
    with pytest.raises(RuntimeError, match="dtype"):
        _ = scaler(torch.as_tensor([[3.0, 10.0]], dtype=torch.float64))


def test_select_device_with_injected_availability():
    assert core_runtime.select_device("cpu").type == "cpu"
    assert (
        core_runtime.select_device(
            "auto",
            cuda_is_available_fn=lambda: False,
            mps_is_available_fn=lambda: False,
        ).type
        == "cpu"
    )
    assert (
        core_runtime.select_device(
            "auto",
            cuda_is_available_fn=lambda: True,
            mps_is_available_fn=lambda: False,
        ).type
        == "cuda"
    )
    assert (
        core_runtime.select_device(
            "auto",
            cuda_is_available_fn=lambda: False,
            mps_is_available_fn=lambda: True,
        ).type
        == "mps"
    )
    with pytest.raises(ValueError, match="CUDA is not available"):
        _ = core_runtime.select_device(
            "cuda",
            cuda_is_available_fn=lambda: False,
            mps_is_available_fn=lambda: False,
        )
    with pytest.raises(ValueError, match="MPS is not available"):
        _ = core_runtime.select_device(
            "mps",
            cuda_is_available_fn=lambda: False,
            mps_is_available_fn=lambda: False,
        )


def test_obs_scale_from_env_infinite_bounds():
    env_conf = SimpleNamespace(
        gym_conf=SimpleNamespace(
            transform_state=True,
            state_space=SimpleNamespace(
                low=np.asarray([-np.inf, -np.inf], dtype=np.float32),
                high=np.asarray([np.inf, np.inf], dtype=np.float32),
                shape=(2,),
            ),
        ),
        ensure_spaces=lambda: None,
    )
    lb, width = core_runtime.obs_scale_from_env(env_conf)
    assert np.allclose(lb, np.asarray([0.0, 0.0], dtype=np.float32))
    assert np.allclose(width, np.asarray([1.0, 1.0], dtype=np.float32))


def test_seed_everything_reproducible():
    core_runtime.seed_everything(123, cuda_is_available_fn=lambda: False)
    np_a = float(np.random.rand())
    torch_a = float(torch.rand(1).item())
    core_runtime.seed_everything(123, cuda_is_available_fn=lambda: False)
    np_b = float(np.random.rand())
    torch_b = float(torch.rand(1).item())
    assert np_a == np_b
    assert torch_a == torch_b
