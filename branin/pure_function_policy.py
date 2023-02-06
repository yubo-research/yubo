import numpy as np


class PureFunctionPolicy:
    def __init__(self, env_conf):
        self._env_conf = env_conf
        self._params = np.random.uniform(-1, 1, size=(len(env_conf.action_space.low),))

    def num_params(self):
        return len(self._params)

    def set_params(self, x):
        self._params = x.copy()

    def get_params(self):
        return self._params

    def clone(self):
        pfp = PureFunctionPolicy(self._env_conf)
        pfp._params = self._params.copy()
        return pfp

    def __call__(self, state):
        assert state == 0, state
        return self._params
