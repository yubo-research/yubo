import numpy as np
from scipy.stats import qmc


class NoiseMaker:
    def __init__(self, env, normalized_noise_level, num_measurements=1000, seed=17):
        self._env = env
        self._real_noise_level = normalized_noise_level * self._measure_noise(env, num_measurements)
        self._rng = np.random.default_rng(seed)

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        ret = list(self._env.step(action))
        ret[1] += self._real_noise_level * self._rng.normal()
        return tuple(ret)

    def reset(self, seed):
        return self._env.reset(seed)

    def close(self):
        self._env.close()

    def _measure_noise(self, env, num_measurements):
        y = []
        actions = qmc.Sobol(env.action_space.shape[0], seed=17).random(num_measurements)
        for action in actions:
            y.append(env.step(action)[1])
        return np.std(y)
