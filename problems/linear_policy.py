import numpy as np

from .normalizer import Normalizer


# Simple random search of static linear policies is competitive for reinforcement learning
# https://proceedings.neurips.cc/paper_files/paper/2018/hash/7634ea65a4e6d9041cfd3f7de18e334a-Abstract.html
# Ant: 4000 after ~50,000 episodes
class LinearPolicy:
    def __init__(self, env_conf):
        self.problem_seed = env_conf.problem_seed
        self._env_conf = env_conf
        num_state = env_conf.gym_conf.state_space.shape[0]
        self._beta = np.random.uniform(
            -1,
            1,
            size=(
                env_conf.action_space.shape[0],
                num_state,
            ),
        )
        self._normalizer = Normalizer(shape=(num_state,))
        self._num_beta = self._beta.size
        self._scale = 1

    def num_params(self):
        return self._num_beta + 1

    def set_params(self, x):
        # x in [-1,1]
        assert x.min() >= -1 and x.max() <= 1, (x.min(), x.max())
        i = 0
        self._scale = x[0]
        i += 1
        self._beta = x[i : i + self._num_beta].reshape(self._beta.shape)

    def get_params(self):
        p = np.zeros(shape=(self.num_params(),))
        i = 0
        p[0] = self._scale
        i += 1
        p[i : i + self._num_beta] = self._beta.flatten()
        return p

    def clone(self):
        lp = LinearPolicy(self._env_conf)
        lp._beta = self._beta.copy()
        return lp

    def __call__(self, state):
        # beta in [-1, 1]
        # scale in [-1, 1]
        assert self._beta.min() >= -1 and self._beta.max() <= 1, (self._beta.min(), self._beta.max())
        self._normalizer.update(state)
        loc = self._normalizer.mean()
        scale = self._normalizer.std()
        i = np.where(scale == 0)[0]
        scale[i] = 1
        state = (state - loc) / scale
        state[i] = 0.0
        k = 10**self._scale  # k in [0.1, 10]
        beta = k * self._beta
        return np.maximum(-1, np.minimum(1, beta @ state))
