from types import SimpleNamespace

import numpy as np
import torch

from rl.algos import torchrl_common


def test_temporary_distribution_validate_args_restores_after_exception():
    previous = getattr(torch.distributions.Distribution, "_validate_args", True)
    try:
        with torchrl_common.temporary_distribution_validate_args(False):
            assert getattr(torch.distributions.Distribution, "_validate_args", None) is False
            raise RuntimeError("boom")
    except RuntimeError as exc:
        assert "boom" in str(exc)
    assert getattr(torch.distributions.Distribution, "_validate_args", None) is previous


def test_collector_device_kwargs_routes_env_to_cpu():
    policy_device = torch.device("cpu")
    kwargs = torchrl_common.collector_device_kwargs(policy_device)
    assert kwargs["env_device"].type == "cpu"
    assert kwargs["policy_device"] == policy_device
    assert kwargs["storing_device"] == policy_device


def test_obs_scaler_applies_affine_transform():
    scaler = torchrl_common.ObsScaler(
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([2.0, 4.0], dtype=np.float32),
    )
    obs = torch.tensor([[3.0, 10.0]], dtype=torch.float32)
    scaled = scaler(obs)
    assert torch.allclose(scaled, torch.tensor([[1.0, 2.0]], dtype=torch.float32))


def test_select_device_respects_cpu_and_auto_fallback(monkeypatch):
    assert torchrl_common.select_device("cpu").type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert torchrl_common.select_device("auto").type == "cpu"


def test_obs_scale_from_env_handles_infinite_bounds():
    gym_conf = SimpleNamespace(
        transform_state=True,
        state_space=SimpleNamespace(
            low=np.array([-np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf], dtype=np.float32),
            shape=(2,),
        ),
    )
    env_conf = SimpleNamespace(gym_conf=gym_conf, ensure_spaces=lambda: None)
    lb, width = torchrl_common.obs_scale_from_env(env_conf)
    assert np.allclose(lb, np.zeros((2,), dtype=np.float32))
    assert np.allclose(width, np.ones((2,), dtype=np.float32))
