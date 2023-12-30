import numpy as np

from .ms_filter import MeanStdFilter


class LinearPolicy:
    def __init__(self, env_conf):
        self.problem_seed = env_conf.problem_seed
        self._env_conf = env_conf
        num_state = env_conf.state_space.shape[0]
        self._beta = np.random.uniform(
            -1,
            1,
            size=(
                env_conf.action_space.shape[0],
                num_state,
            ),
        )
        self._ms_filter = MeanStdFilter(env_conf.state_space.shape)

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
        # beta in [-1, 1]
        # state in [0,1]
        self._ms_filter(state)
        m, s = self._ms_filter.get_stats()
        state = state - m
        state = state / s
        return np.tanh(self._beta @ state)
