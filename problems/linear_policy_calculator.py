import copy

import numpy as np

from problems.normalizer import Normalizer


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

        # We recreate the normalizer every time b/c we want this policy to
        #  be clonable and we want our results to be reproducible.
        # Could we replace the normalizer with parameters for loc and scale?
        self._normalizer = None
        self._num_beta = self._beta.size
        self._scale = 1
        self._num_state = num_state
        self._loc_0 = np.random.uniform(-1, 1, size=(self._num_state,))
        self._scale_0 = np.random.uniform(0, 1, size=(self._num_state,))
        self._num_init_x = np.random.uniform(0, 1)
        self._K = 3
        self._max_num_init_x = 10

    def num_params(self):
        return self._num_beta + 2 + 2 * self._num_state

    def set_params(self, x):
        assert x.min() >= -1 and x.max() <= 1, (x.min(), x.max())
        i = 0
        self._scale = x[0]
        i += 1
        self._beta = x[i : i + self._num_beta].reshape(self._beta.shape)
        i += self._num_beta
        self._loc_0 = self._K * x[i : i + self._num_state]
        i += self._num_state
        self._scale_0 = self._K * (0.5 * (1 + x[i : i + self._num_state]))
        i += self._num_state
        self._num_init_x = (1 + x[i]) / 2
        i += 1
        self._normalizer = Normalizer(
            shape=(self._num_state,),
            num_init=int(self._max_num_init_x * self._num_init_x),
            init_mean=self._loc_0,
            init_var=self._scale_0,
        )

        self._k = 1 * (1 + self._scale)

    def get_params(self):
        p = np.zeros(shape=(self.num_params(),))
        i = 0
        p[0] = self._scale
        i += 1
        p[i : i + self._num_beta] = self._beta.flatten()
        i += self._num_beta
        p[i : i + self._num_state] = self._loc_0 / self._K
        i += self._num_state
        p[i : i + self._num_state] = 2 * self._scale_0 / self._K - 1
        i += self._num_state
        p[i] = 2 * self._num_init_x - 1
        i += 1
        return p

    def clone(self):
        calc = LinearPolicyCalculator(self._id_int, self._num_state, self._num_action)
        calc._beta = self._beta.copy()
        calc._scale = self._scale
        calc._normalizer = copy.deepcopy(self._normalizer)
        if hasattr(self, "_k"):
            calc._k = self._k
        return calc

    def _normalize(self, state):
        self._normalizer.update(state)
        loc, scale = self._normalizer.mean_and_std()
        # print("LS:", loc, scale)

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
