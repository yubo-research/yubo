import numpy as np

from problems.normalizer import Normalizer

_normalizer = {}


class LinearPolicyCalculator:
    def __init__(self, id_int, num_state, num_action):
        self._id_int = id_int
        self._num_state = num_state
        self._num_action = num_action
        self._beta = np.random.uniform(
            -1,
            1,
            size=(
                num_action,
                num_state,
            ),
        )

        if id_int not in _normalizer:
            _normalizer[id_int] = Normalizer(shape=(num_state,))
        self._normalizer = _normalizer[id_int]
        self._num_beta = self._beta.size
        self._scale = 1

    def num_params(self):
        return self._num_beta + 1

    def set_params(self, x):
        assert x.min() >= -1 and x.max() <= 1, (x.min(), x.max())
        i = 0
        self._scale = x[0]
        i += 1
        self._beta = x[i : i + self._num_beta].reshape(self._beta.shape)
        self._k = 2 * (1 + self._scale)

    def get_params(self):
        p = np.zeros(shape=(self.num_params(),))
        i = 0
        p[0] = self._scale
        i += 1
        p[i : i + self._num_beta] = self._beta.flatten()
        return p

    def clone(self):
        calc = LinearPolicyCalculator(self._id_int, self._num_state, self._num_action)
        calc._beta = self._beta.copy()
        calc._scale = self._scale
        calc._normalizer = self._normalizer
        if hasattr(self, "_k"):
            calc._k = self._k
        return calc

    def _normalize(self, state):
        self._normalizer.update(state)
        loc, scale = self._normalizer.mean_and_std()
        # print("LS:", np.abs(loc).mean(), np.abs(scale).mean())

        state = state - loc

        i = np.where(scale == 0)[0]
        scale[i] = 1
        state[i] = 0.0
        state = state / scale

        return state

    def calculate(self, state):
        assert self._beta.min() >= -1 and self._beta.max() <= 1, (self._beta.min(), self._beta.max())
        state = self._normalize(state)
        beta = self._k * self._beta

        return np.maximum(-1, np.minimum(1, beta @ state))
