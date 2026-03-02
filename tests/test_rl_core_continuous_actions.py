import numpy as np
import pytest
import torch

from rl.core.continuous_actions import (
    normalize_action_bounds,
    scale_action_tensor_to_env,
    scale_action_to_env,
    unscale_action_from_env,
    unscale_action_tensor_from_env,
)


def test_normalize_action_bounds_scalar_expand_and_sentinel():
    f32_max = float(np.finfo(np.float32).max)
    low = np.asarray([-f32_max], dtype=np.float32)
    high = np.asarray([f32_max], dtype=np.float32)
    low_n, high_n = normalize_action_bounds(low, high, dim=3)
    assert low_n.shape == (3,)
    assert high_n.shape == (3,)
    assert np.all(low_n == -1.0)
    assert np.all(high_n == 1.0)


def test_normalize_action_bounds_raises_on_dim_mismatch():
    with pytest.raises(ValueError, match="Action bounds must match action dimension"):
        _ = normalize_action_bounds(np.asarray([0.0, 1.0]), np.asarray([1.0]), dim=3)


def test_scale_and_unscale_round_trip():
    low = np.asarray([-2.0, 0.0], dtype=np.float32)
    high = np.asarray([2.0, 4.0], dtype=np.float32)
    action_unit = np.asarray([[-1.0, 0.5]], dtype=np.float32)
    action_env = scale_action_to_env(action_unit, low, high, clip=True)
    recovered = unscale_action_from_env(action_env, low, high, clip=True)
    assert np.allclose(action_unit, recovered, atol=1e-6)


def test_scale_clip_controls_bounds():
    low = np.asarray([-1.0], dtype=np.float32)
    high = np.asarray([1.0], dtype=np.float32)
    action_unit = np.asarray([[2.0]], dtype=np.float32)
    clipped = scale_action_to_env(action_unit, low, high, clip=True)
    unclipped = scale_action_to_env(action_unit, low, high, clip=False)
    assert np.all(clipped <= 1.0)
    assert np.any(unclipped > 1.0)


def test_unscale_action_tensor_from_env():
    low = torch.as_tensor([-2.0, 0.0], dtype=torch.float32)
    high = torch.as_tensor([2.0, 4.0], dtype=torch.float32)
    action = torch.as_tensor([[-2.0, 2.0], [4.0, 5.0]], dtype=torch.float32)

    clipped = unscale_action_tensor_from_env(action, low, high, clip=True)
    unclipped = unscale_action_tensor_from_env(action, low, high, clip=False)

    assert torch.all(clipped <= 1.0)
    assert torch.all(clipped >= -1.0)
    assert torch.any(unclipped > 1.0)


def test_scale_action_tensor_to_env_round_trip():
    low = torch.as_tensor([-2.0, 0.0], dtype=torch.float32)
    high = torch.as_tensor([2.0, 4.0], dtype=torch.float32)
    action_unit = torch.as_tensor([[-1.0, 0.5], [1.5, -2.0]], dtype=torch.float32)

    clipped_env = scale_action_tensor_to_env(action_unit, low, high, clip=True)
    unclipped_env = scale_action_tensor_to_env(action_unit, low, high, clip=False)
    recovered = unscale_action_tensor_from_env(clipped_env, low, high, clip=True)

    assert torch.all(clipped_env <= high)
    assert torch.all(clipped_env >= low)
    assert torch.any(unclipped_env > high)
    assert torch.all(recovered <= 1.0)
    assert torch.all(recovered >= -1.0)
