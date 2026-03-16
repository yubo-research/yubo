"""Tests for rl.sac.runtime_utils."""

from types import SimpleNamespace

import numpy as np

from rl.sac import runtime_utils


def test_select_device_cpu():
    dev = runtime_utils.select_device("cpu")
    assert dev.type == "cpu"


def test_obs_scale_from_env():
    env_conf = SimpleNamespace(
        gym_conf=SimpleNamespace(
            transform_state=True,
            state_space=SimpleNamespace(
                low=np.asarray([-1.0, 0.0], dtype=np.float32),
                high=np.asarray([1.0, 2.0], dtype=np.float32),
                shape=(2,),
            ),
        ),
        ensure_spaces=lambda: None,
    )
    lb, width = runtime_utils.obs_scale_from_env(env_conf)
    assert lb.shape == (2,)
    assert width.shape == (2,)
    assert np.allclose(lb, np.array([-1.0, 0.0], dtype=np.float32))
    assert np.allclose(width, np.array([2.0, 2.0], dtype=np.float32))
