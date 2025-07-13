from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from kheperax.tasks.target import TargetKheperaxTask

from problems.env_conf import GymConf

_wrappers = {}


@dataclass
class KheperaxEnvConf:
    kheperax_config: any
    problem_seed: int = None
    max_steps: int = 1000
    noise_seed_0: int = 0

    def __post_init__(self):
        self.gym_conf = GymConf(max_steps=self.max_steps)
        # Initialize state_space with a default size (will be updated in make())
        self.gym_conf.state_space = MockStateSpace(5)  # Default kheperax observation size
        # Initialize action_space with a default size (will be updated in make())
        self.action_space = MockActionSpace(2)  # Default kheperax action size
        self.env_name = "kheperax"
        self.policy_class = None
        self.noise_level = None
        self.frozen_noise = True
        self.kwargs = {}

    def _init_action_space(self, env):
        action_size = env.action_size
        self.action_space = MockActionSpace(action_size)

    def make(self, **kwargs):
        key = f"{self.problem_seed}-{kwargs}-{repr(self.kheperax_config)}"
        if key not in _wrappers:
            print("MAKING KHEPERAX ENVIRONMENT", key)
            env, policy_network, scoring_fn = TargetKheperaxTask.create_default_task(
                kheperax_config=self.kheperax_config,
                random_key=jax.random.PRNGKey(self.problem_seed),
            )
            self._init_action_space(env)
            self._init_gym_conf(env)
            _wrappers[key] = KheperaxGymWrapper(env, self.max_steps)
        return _wrappers[key]

    def _init_gym_conf(self, env):
        # Set up gym_conf with state_space for LinearPolicy compatibility
        if not hasattr(self.gym_conf, "state_space") or self.gym_conf.state_space is None:
            obs_size = env.observation_size
            self.gym_conf.state_space = MockStateSpace(obs_size)

    def close(self):
        pass


class MockActionSpace:
    def __init__(self, action_size):
        self.action_size = action_size
        self.low = np.array([-1.0] * action_size)
        self.high = np.array([1.0] * action_size)

    @property
    def shape(self):
        return (self.action_size,)


class KheperaxGymWrapper:
    def __init__(self, kheperax_env, max_steps):
        self.env = kheperax_env
        self.max_steps = max_steps
        self.current_state = None
        self.step_count = 0
        obs_size = self.env.observation_size
        self.observation_space = MockObservationSpace(obs_size)
        action_size = self.env.action_size
        self.action_space = MockActionSpace(action_size)

    def reset(self, seed=None):
        rng = jax.random.PRNGKey(seed if seed is not None else 0)
        self.current_state = self.env.reset(rng)
        self.step_count = 0
        obs = np.array(self.current_state.obs)
        return obs, {}

    def step(self, action):
        action = jnp.array(action)
        self.current_state = self.env.step(self.current_state, action)
        self.step_count += 1
        obs = np.array(self.current_state.obs)
        reward = float(self.current_state.reward)
        done = bool(self.current_state.done)
        if self.step_count >= self.max_steps:
            done = True
        info = {}
        print("STEP", self.step_count, reward, done)
        return obs, reward, done, False, info

    def close(self):
        pass

    def render(self):
        return np.zeros((64, 64, 3), dtype=np.uint8)


class MockObservationSpace:
    def __init__(self, obs_size):
        self.obs_size = obs_size
        self.low = np.array([-10.0] * obs_size)
        self.high = np.array([10.0] * obs_size)


class MockStateSpace:
    def __init__(self, obs_size):
        self.shape = (obs_size,)
