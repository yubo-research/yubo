import numpy as np


class LinearPolicy:
    def __init__(self, env_conf):
        self._env_conf = env_conf
        num_state = env_conf.state_space.shape[0]
        self._x_center = np.zeros(shape=(num_state,))
        self._x_scale = np.zeros(shape=(num_state,))
        self._beta = np.random.uniform(
            -1,
            1,
            size=(
                env_conf.action_space.shape[0],
                num_state,
            ),
        )
        self._set_k()

    def _set_k(self):
        self._k_scale = 10 * (1 + self._x_scale) / 2

    def num_params(self):
        return 2 * len(self._x_scale) + self._beta.size

    def set_params(self, x):
        n = len(self._x_scale)
        self._x_center = x[:n]
        self._x_scale = x[n : 2 * n]
        self._beta = x[2 * n :].reshape(self._beta.shape)
        self._set_k()

    def get_params(self):
        return np.concatenate((self._x_center, self._x_scale, self._beta.flatten()))

    def clone(self):
        lp = LinearPolicy(self._env_conf)
        lp._x_center = self._x_center.copy()
        lp._x_scale = self._x_scale.copy()
        lp._beta = self._beta.copy()
        lp._set_k()
        return lp

    def __call__(self, state):
        state = state.T - self._x_center
        state = self._k_scale * state
        return np.tanh(self._beta @ state.T)


# lp = LinearPolicy(env_conf)
# b_0 = lp._beta.copy()
# lp.set_params(lp.get_params())
# assert np.all(b_0 == lp._beta)
