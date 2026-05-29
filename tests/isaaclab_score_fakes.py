from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from gymnasium import spaces


class FakeIsaacEnv:
    observation_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self) -> None:
        self._step = 0

    def reset(self, *, seed=None, **_kwargs):
        self._step = 0
        value = 0.0 if seed is None else float(int(seed) % 3)
        return np.asarray([value, 1.0], dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        a = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        obs = np.asarray([float(self._step), a], dtype=np.float32)
        reward = 1.0 - abs(a)
        terminated = self._step >= 3
        return obs, reward, terminated, False, {}

    def close(self):
        return None


def make_fake_runtime():
    runtime = SimpleNamespace(
        env_name="isaaclab:Fake-v0",
        env_tag="isaaclab:Fake-v0",
        problem_seed=0,
        noise_seed_0=0,
        frozen_noise=True,
        state_space=FakeIsaacEnv.observation_space,
        action_space=FakeIsaacEnv.action_space,
        gym_conf=SimpleNamespace(max_steps=3, num_frames_skip=1, transform_state=False),
    )
    runtime.ensure_spaces = lambda: None
    runtime.make = lambda **_kwargs: FakeIsaacEnv()
    return runtime


def make_fake_dm_control_runtime():
    runtime = make_fake_runtime()
    runtime.env_name = "dm_control/quadruped-run-v0"
    runtime.env_tag = "dm_control/quadruped-run-v0"
    return runtime


class FakeVectorIsaacEnv:
    observation_space = FakeIsaacEnv.observation_space
    action_space = FakeIsaacEnv.action_space

    def __init__(self, num_envs: int) -> None:
        self.num_envs = int(num_envs)
        self._envs = [FakeIsaacEnv() for _ in range(self.num_envs)]

    def reset_batch(self, *, seed=None, **_kwargs):
        obs = np.stack([env.reset(seed=seed)[0] for env in self._envs], axis=0)
        return obs, {}

    def step_batch(self, actions):
        actions = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, -1)
        obs_list = []
        rewards = []
        terms = []
        truncs = []
        for idx, env in enumerate(self._envs):
            obs, reward, term, trunc, _info = env.step(actions[idx])
            obs_list.append(obs)
            rewards.append(reward)
            terms.append(term)
            truncs.append(trunc)
        return (
            np.stack(obs_list, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terms, dtype=bool),
            np.asarray(truncs, dtype=bool),
            {},
        )

    def close(self):
        for env in self._envs:
            env.close()


def make_fake_vector_runtime():
    runtime = make_fake_runtime()
    runtime.vector_slots = []

    def _make(**kwargs):
        if kwargs.get("batched"):
            slots = int(kwargs["num_envs"])
            runtime.vector_slots.append(slots)
            return FakeVectorIsaacEnv(slots)
        return FakeIsaacEnv()

    runtime.make = _make
    return runtime
