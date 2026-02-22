"""Tests for dm_control envs via Shimmy (problems/shimmy_dm_control)."""

import gymnasium as gym
import numpy as np
import pytest

from problems.shimmy_dm_control import (
    _flatten_obs,
    _parse_env_name,
    _PixelObsWrapper,
    make,
)


def test_get_env_conf_dm_control():
    """get_env_conf accepts dm: and dm_control/ tags."""
    import problems.env_conf_atari_dm  # noqa: F401 - register atari/dm handlers
    from problems.env_conf import get_env_conf

    ec = get_env_conf("dm:cheetah-run", problem_seed=42, noise_seed_0=100)
    assert ec.env_name == "dm_control/cheetah-run-v0"
    assert ec.problem_seed == 42
    assert ec.noise_seed_0 == 100
    assert ec.gym_conf is not None
    assert ec.gym_conf.transform_state is False

    ec2 = get_env_conf("dm_control/hopper-hop-v0", problem_seed=1)
    assert ec2.env_name == "dm_control/hopper-hop-v0"
    assert ec2.problem_seed == 1


def test_parse_env_name():
    domain, task = _parse_env_name("dm_control/cheetah-run-v0")
    assert domain == "cheetah"
    assert task == "run"

    domain, task = _parse_env_name("dm_control/hopper-hop-v1")
    assert domain == "hopper"
    assert task == "hop"


def test_parse_env_name_invalid_raises():
    with pytest.raises(ValueError, match="Expected dm_control env name"):
        _parse_env_name("gymnasium/HalfCheetah-v5")
    with pytest.raises(ValueError, match="dm_control/cheetah"):
        _parse_env_name("dm_control/cheetah")


def test_flatten_obs():
    flat = _flatten_obs({"a": np.array([1.0, 2.0]), "b": np.array([3.0])})
    np.testing.assert_array_equal(flat, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    flat_scalar = _flatten_obs(np.array(5.0))
    np.testing.assert_array_equal(flat_scalar, np.array([5.0], dtype=np.float32))
    flat_empty = _flatten_obs({})
    assert flat_empty.shape == (0,)
    assert flat_empty.dtype == np.float32


def test_dm_control_cheetah_run_real():
    """Integration test: make() creates env, reset/step work, seeding is deterministic."""
    pytest.importorskip("dm_control")
    pytest.importorskip("shimmy")
    env = make("dm_control/cheetah-run-v0")
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 1
    assert obs.dtype == np.float32
    assert obs.size == env.observation_space.shape[0]

    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    obs2, reward, terminated, truncated, extra = env.step(action)
    assert isinstance(obs2, np.ndarray)
    assert isinstance(reward, (int, float))

    env2 = make("dm_control/cheetah-run-v0")
    obs3, _ = env2.reset(seed=42)
    env2.close()
    np.testing.assert_array_equal(obs, obs3)

    env.close()


def test_dm_control_from_pixels():
    """Integration test: from_pixels returns pixel obs."""
    pytest.importorskip("dm_control")
    pytest.importorskip("shimmy")
    env = make("dm_control/cheetah-run-v0", from_pixels=True, pixels_only=True)
    obs, info = env.reset(seed=42)
    assert isinstance(obs, dict)
    assert "pixels" in obs
    assert obs["pixels"].shape == (84, 84, 3)
    assert obs["pixels"].dtype == np.uint8
    env.close()


def test_pixel_wrapper_contract_pixels_only_is_dict():
    class _DummyEnv(gym.Env):
        metadata = {"render_modes": ["rgb_array"]}

        def __init__(self):
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            _ = options
            super().reset(seed=seed)
            return np.zeros((3,), dtype=np.float32), {}

        def step(self, action):
            _ = action
            return np.zeros((3,), dtype=np.float32), 0.0, False, False, {}

        def render(self):
            return np.zeros((84, 84, 3), dtype=np.uint8)

    wrapped = _PixelObsWrapper(_DummyEnv(), pixels_only=True, size=84)
    assert isinstance(wrapped.observation_space, gym.spaces.Dict)
    assert "pixels" in wrapped.observation_space.spaces
    obs, _ = wrapped.reset(seed=0)
    assert isinstance(obs, dict)
    assert obs["pixels"].shape == (84, 84, 3)
