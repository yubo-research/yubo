import numpy as np
import torch
from tensordict import TensorDict

from rl.torchrl.offpolicy.trainer_utils import normalize_actions_for_replay
from rl.torchrl.sac.trainer import _scale_action_to_env, _unscale_action_from_env


def test_scale_action_to_env():
    low = np.array([-2.0], dtype=np.float32)
    high = np.array([4.0], dtype=np.float32)
    np.testing.assert_allclose(_scale_action_to_env(np.array([-1.0]), low, high), [-2.0])
    np.testing.assert_allclose(_scale_action_to_env(np.array([1.0]), low, high), [4.0])
    np.testing.assert_allclose(_scale_action_to_env(np.array([0.0]), low, high), [1.0])


def test_unscale_action_from_env():
    low = np.array([-2.0], dtype=np.float32)
    high = np.array([4.0], dtype=np.float32)
    np.testing.assert_allclose(_unscale_action_from_env(np.array([-2.0]), low, high), [-1.0])
    np.testing.assert_allclose(_unscale_action_from_env(np.array([4.0]), low, high), [1.0])
    np.testing.assert_allclose(_unscale_action_from_env(np.array([1.0]), low, high), [0.0])


def test_normalize_actions_for_replay_uses_unit_range():
    flat = TensorDict(
        {"action": torch.as_tensor([[-2.0], [1.0], [4.0]], dtype=torch.float32)},
        batch_size=[3],
    )
    out = normalize_actions_for_replay(
        flat,
        action_low=np.array([-2.0], dtype=np.float32),
        action_high=np.array([4.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        out["action"].detach().cpu().numpy().reshape(-1),
        np.asarray([-1.0, 0.0, 1.0], dtype=np.float32),
        atol=1e-6,
    )
