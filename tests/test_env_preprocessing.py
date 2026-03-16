from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from common.env_preprocessing import (
    EnvPreprocessingSpec,
    _ClipRewardWrapper,
    apply_gym_preprocessing,
    attach_env_preprocessing,
    spec_from_config,
    spec_from_env_conf,
)


class _DummyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        _ = seed, options
        return np.asarray([20.0, -20.0], dtype=np.float32), {}

    def step(self, action):
        _ = action
        obs = np.asarray([30.0, -30.0], dtype=np.float32)
        reward = 50.0
        terminated = True
        truncated = False
        return obs, reward, terminated, truncated, {}


def test_spec_from_config_parses_clip_fields():
    cfg = SimpleNamespace(
        normalize_observation=True,
        normalize_reward=False,
        reward_normalize_gamma=0.97,
        observation_clip=[-10.0, 10.0],
        reward_clip=5.0,
    )
    spec = spec_from_config(cfg)
    assert spec.normalize_observation is True
    assert spec.normalize_reward is False
    assert spec.reward_normalize_gamma == 0.97
    assert spec.observation_clip == (-10.0, 10.0)
    assert spec.reward_clip == (-5.0, 5.0)


def test_apply_gym_preprocessing_clips_observation_and_reward():
    env = _DummyEnv()
    spec = EnvPreprocessingSpec(
        normalize_observation=False,
        normalize_reward=False,
        observation_clip=(-10.0, 10.0),
        reward_clip=(-5.0, 5.0),
    )
    wrapped = apply_gym_preprocessing(env, preprocess_spec=spec)

    obs0, _ = wrapped.reset(seed=0)
    assert np.allclose(obs0, np.asarray([10.0, -10.0], dtype=np.float32))
    obs1, reward, terminated, truncated, _ = wrapped.step(np.asarray([0.0], dtype=np.float32))
    assert np.allclose(obs1, np.asarray([10.0, -10.0], dtype=np.float32))
    assert reward == 5.0
    assert terminated is True
    assert truncated is False


def test_attach_and_read_back_spec_from_env_conf():
    env_conf = SimpleNamespace()
    spec = EnvPreprocessingSpec(normalize_observation=True, normalize_reward=True)
    attach_env_preprocessing(env_conf, spec)
    restored = spec_from_env_conf(env_conf)
    assert restored == spec


def test_enabled_property_and_noop_apply():
    env = _DummyEnv()
    off = EnvPreprocessingSpec()
    on = EnvPreprocessingSpec(normalize_observation=True)
    assert off.enabled is False
    assert on.enabled is True
    wrapped = apply_gym_preprocessing(env, preprocess_spec=off)
    assert wrapped is env


def test_clip_reward_wrapper_reward_method():
    env = _DummyEnv()
    wrapper = _ClipRewardWrapper(env, low=-2.0, high=2.0)
    assert wrapper.reward(10.0) == 2.0


def test_clip_observation_wrapper_dict_and_tuple_obs():
    from common.env_preprocessing import _ClipObservationWrapper

    class _DictObsEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Dict(
                a=gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                b=gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            )
            self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return {"a": np.array([50.0]), "b": np.array([-50.0])}, {}

        def step(self, action):
            return {"a": np.array([30.0]), "b": np.array([-30.0])}, 0.0, True, False, {}

    env = _DictObsEnv()
    wrapped = _ClipObservationWrapper(env, low=-10.0, high=10.0)
    obs, _ = wrapped.reset(seed=0)
    assert np.allclose(obs["a"], np.array([10.0]))
    assert np.allclose(obs["b"], np.array([-10.0]))
    obs, _, term, trunc, _ = wrapped.step(np.array([0.0]))
    assert np.allclose(obs["a"], np.array([10.0]))
    assert term and not trunc
