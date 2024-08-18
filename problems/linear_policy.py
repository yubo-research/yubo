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
        self._center = np.random.uniform(-1, 1, size=(num_state,))
        self._scale = np.random.uniform()

        self._num_beta = self._beta.size
        self._num_center = self._center.size

    def num_params(self):
        return 1 + self._num_beta + self._num_center

    def set_params(self, x):
        self._scale = x[0]
        self._beta = x[1 : 1 + self._num_beta].reshape(self._beta.shape)
        self._center = x[1 + self._num_beta :].reshape(self._center.shape)

    def get_params(self):
        p = np.zeros(shape=(self.num_params(),))
        p[0] = self._scale
        p[1 : 1 + self._num_beta] = self._beta.flatten()
        p[1 + self._num_beta :] = self._center.flatten()
        return p

    def clone(self):
        lp = LinearPolicy(self._env_conf)
        lp._scale = self._scale
        lp._beta = self._beta.copy()
        lp._center = self._center.copy()
        return lp

    def __call__(self, state):
        # beta in [-1, 1]
        # state in [0,1]
        state = state - self._center

        scale = 10 * np.abs(self._scale)
        # scale = 10 * np.exp(self._scale)
        norm = np.sqrt((self._beta**2).sum())
        if norm > 0:
            beta = scale * self._beta / norm
        else:
            beta = 0.0 * self._beta
        return np.maximum(-1, np.minimum(1, beta @ state))
