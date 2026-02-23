import numpy as np

from rl.backends.torchrl.sac.trainer import (
    _scale_action_to_env,
    _unscale_action_from_env,
)


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
