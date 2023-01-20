import numpy as np


class LinearPolicy:
    def __init__(self, env_conf):
        self._env_conf = env_conf
        self._x_state = 0
        self._beta = np.random.uniform(
            -1,
            1,
            size=(
                env_conf.action_space.shape[0],
                env_conf.state_space.shape[0],
            ),
        )
        self._set_k()

    def _set_k(self):
        self._k_state = 10*(1 + self._x_state)/2
        
    def num_params(self):
        return 1 + self._beta.size

    def set_params(self, x):
        self._x_state = x[0]
        self._beta = x[1:].reshape(self._beta.shape)
        self._set_k()

    def get_params(self):
        x = [self._x_state] + list(self._beta.flatten())
        return np.array(x)

    def clone(self):
        lp = LinearPolicy(self._env_conf)
        lp._x_state = self._x_state
        lp._beta = self._beta.copy()
        lp._set_k()
        return lp

    def __call__(self, state):
        state = self._k_state * state
        return np.tanh(self._beta @ state)


# lp = LinearPolicy(env_conf)
# b_0 = lp._beta.copy()
# lp.set_params(lp.get_params())
# assert np.all(b_0 == lp._beta)
