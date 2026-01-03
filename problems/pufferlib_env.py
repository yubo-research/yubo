import numpy as np


class PufferLibBreakoutEnv:
    def __init__(self, num_envs=1):
        from pufferlib.ocean.breakout.breakout import Breakout

        self._env = Breakout(num_envs=num_envs)
        self._num_envs = num_envs
        self.observation_space = self._env.single_observation_space
        self.action_space = self._env.single_action_space

    def reset(self, seed=None):
        obs, info = self._env.reset(seed=seed)
        if self._num_envs == 1:
            obs = obs[0]
        return obs, info

    def step(self, action):
        if self._num_envs == 1:
            action = np.array([action])
        obs, reward, done, truncated, info = self._env.step(action)
        if self._num_envs == 1:
            obs = obs[0]
            reward = float(reward[0])
            done = bool(done[0])
            truncated = bool(truncated[0]) if truncated is not None else False
        return obs, reward, done, truncated, info

    def close(self):
        self._env.close()

    def render(self):
        return None
