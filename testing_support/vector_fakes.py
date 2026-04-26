"""Vector env test doubles for Puffer API tests."""

from types import SimpleNamespace

import numpy as np


class FakePufferVecEnv:
    def __init__(self, num_envs: int):
        self.num_envs = int(num_envs)
        self.single_action_space = SimpleNamespace(n=6)
        self.single_observation_space = SimpleNamespace(shape=(80, 4, 105))
        self._t = 0

    def reset(self, seed=None):
        _ = seed
        self._t = 0
        obs = np.zeros((self.num_envs, 80, 4, 105), dtype=np.uint8)
        infos = [{} for _ in range(self.num_envs)]
        return obs, infos

    def step(self, action):
        _ = action
        self._t += 1
        obs = np.zeros((self.num_envs, 80, 4, 105), dtype=np.uint8)
        rew = np.ones((self.num_envs,), dtype=np.float32)
        term = np.zeros((self.num_envs,), dtype=bool)
        trunc = np.zeros((self.num_envs,), dtype=bool)
        infos = []
        if self._t % 2 == 0:
            infos = [{"episode_return": float(self._t), "episode_length": int(self._t)}]
        return obs, rew, term, trunc, infos

    def close(self):
        return None


class FakePufferVecEnvContinuous:
    def __init__(self, num_envs: int):
        self.num_envs = int(num_envs)
        self.single_action_space = SimpleNamespace(
            shape=(4,),
            low=-np.ones((4,), dtype=np.float32),
            high=np.ones((4,), dtype=np.float32),
        )
        self.single_observation_space = SimpleNamespace(shape=(24,))
        self._t = 0

    def reset(self, seed=None):
        _ = seed
        self._t = 0
        obs = np.zeros((self.num_envs, 24), dtype=np.float32)
        infos = [{} for _ in range(self.num_envs)]
        return obs, infos

    def step(self, action):
        action = np.asarray(action)
        assert action.shape == (self.num_envs, 4)
        self._t += 1
        obs = np.zeros((self.num_envs, 24), dtype=np.float32)
        rew = np.ones((self.num_envs,), dtype=np.float32)
        term = np.zeros((self.num_envs,), dtype=bool)
        trunc = np.zeros((self.num_envs,), dtype=bool)
        infos = []
        if self._t % 2 == 0:
            infos = [{"episode_return": float(self._t), "episode_length": int(self._t)}]
        return obs, rew, term, trunc, infos

    def close(self):
        return None
