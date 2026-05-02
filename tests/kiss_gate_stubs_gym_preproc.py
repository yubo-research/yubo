import gymnasium as gym
import numpy as np


class _Space:
    low = np.array([-1.0])
    high = np.array([1.0])


class _Env(gym.Env):
    metadata = {}
    action_space = _Space()
    observation_space = _Space()

    def __init__(self):
        self.k = 0

    def reset(self, seed=None, options=None):
        return np.array([2.0]), {}

    def step(self, _a):
        self.k += 1
        return np.array([3.0]), 1.0, self.k >= 2, False, {}

    def close(self):
        return None
