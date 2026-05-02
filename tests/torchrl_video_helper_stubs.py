from __future__ import annotations

from types import SimpleNamespace

import numpy as np


class _BoxSpaceStub:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


class _EnvStub:
    def __init__(self):
        self.action_space = _BoxSpaceStub(low=[-2.0, -2.0], high=[2.0, 2.0])
        self.actions = []
        self.step_count = 0

    def reset(self, *, seed=None):
        _ = seed
        return np.asarray([0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.step_count += 1
        terminated = self.step_count >= 1
        truncated = False
        return np.asarray([0.0, 0.0], dtype=np.float32), 1.0, terminated, truncated, {}

    def close(self):
        return


class _EnvConfStub:
    def __init__(self, *, max_steps=5, gym_conf=True):
        self.gym_conf = SimpleNamespace(max_steps=max_steps) if gym_conf else None
        self.made_envs = []

    def make(self, **_kwargs):
        env = _EnvStub()
        self.made_envs.append(env)
        return env
