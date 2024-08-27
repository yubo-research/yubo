import numpy as np


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
        self._center = np.random.uniform(0, 1, size=(num_state,))
        self._scale = np.random.uniform(0, 1, size=(num_state,))

        self._num_beta = self._beta.size
        self._num_center = self._center.size
        self._num_scale = self._scale.size

    def num_params(self):
        return 1 + self._num_beta + self._num_center + self._num_scale

    def set_params(self, x):
        # x in [-1,1]
        assert x.min() >= -1 and x.max() <= 1, (x.min(), x.max())
        i = 0
        self._beta = x[i : i + self._num_beta].reshape(self._beta.shape)
        i += self._num_beta
        self._center = (1 + x[i : i + self._num_center].reshape(self._center.shape)) / 2
        i += self._num_center
        self._scale = (1 + x[i : i + self._num_scale].reshape(self._center.shape)) / 2

    def get_params(self):
        p = np.zeros(shape=(self.num_params(),))
        i = 0
        p[i : i + self._num_beta] = self._beta.flatten()
        i += self._num_beta
        p[i : i + self._num_center] = self._center.flatten()
        i += self._num_center
        p[i : i + self._num_scale] = self._scale.flatten()
        return p

    def clone(self):
        lp = LinearPolicy(self._env_conf)
        lp._beta = self._beta.copy()
        lp._center = self._center.copy()
        lp._scale = self._scale.copy()
        return lp

    def __call__(self, state):
        # beta in [-1, 1]
        # state in [0,1]
        state = (state - self._center) / self._scale
        beta = 0.1 * self._beta
        return np.maximum(-1, np.minimum(1, beta @ state))
