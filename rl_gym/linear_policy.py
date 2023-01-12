import numpy as np


class LinearPolicy:
    def __init__(self, env_conf):
        self._env_conf = env_conf
        self._beta = np.random.uniform(
            -1,
            1,
            size=(
                env_conf.action_space.shape[0],
                env_conf.state_space.shape[0],
            ),
        )

    def num_params(self):
        return self._beta.size

    def set_params(self, x):
        self._beta = x.reshape(self._beta.shape)

    def get_params(self):
        return self._beta.flatten()

    def clone(self):
        lp = LinearPolicy(self._env_conf)
        lp._beta = self._beta.copy()
        return lp

    def __call__(self, state):
        state = self._env_conf.k_state * state
        return np.tanh(self._beta @ state)


# lp = LinearPolicy(env_conf)
# b_0 = lp._beta.copy()
# lp.set_params(lp.get_params())
# assert np.all(b_0 == lp._beta)
