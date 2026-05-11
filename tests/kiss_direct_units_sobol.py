from types import SimpleNamespace

import numpy as np


class _SobolEnv:
    action_space = SimpleNamespace(low=np.array([-1.0]), high=np.array([1.0]))

    def __init__(self):
        self.n = 0

    def reset(self, seed=None):
        return np.zeros(1), {}

    def step(self, action):
        self.n += 1
        return np.zeros(1), 1.0, self.n > 1, False, {}

    def close(self):
        return None


class _SobolConf:
    noise_seed_0 = 10
    frozen_noise = False
    gym_conf = SimpleNamespace(max_steps=2)

    @staticmethod
    def make():
        return _SobolEnv()
