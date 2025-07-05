from collections import deque

import numpy as np

from .linear_policy import LinearPolicyCalculator


class ARLinearPolicy:
    def __init__(self, env_conf, num_ar=2, use_differences=False):
        self._env_conf = env_conf
        self._num_ar = num_ar
        self._use_differences = use_differences
        self._queue = deque(maxlen=self._num_ar)
        num_state = env_conf.gym_conf.state_space.shape[0]
        num_action = env_conf.action_space.shape[0]
        if self._use_differences:
            big_state_size = (self._num_ar - 1) * (num_state + num_action + 1)
        else:
            big_state_size = self._num_ar * (num_state + num_action + 1)
        self._calculator = LinearPolicyCalculator(big_state_size, num_action)

    def num_params(self):
        return self._calculator.num_params()

    def set_params(self, x):
        self._calculator.set_params(x)

    def get_params(self):
        return self._calculator.get_params()

    def clone(self):
        ar_policy = ARLinearPolicy(self._env_conf)
        ar_policy._calculator = self._calculator.clone()
        return ar_policy

    def _create_big_state(self):
        if not self._queue:
            return np.zeros(self._calculator._num_state)

        def to_1d_array(x):
            if isinstance(x, np.ndarray):
                return x.ravel()
            return np.array([x])

        expected_state_size = self._env_conf.gym_conf.state_space.shape[0]
        expected_action_size = self._env_conf.action_space.shape[0]
        entry_size = expected_state_size + expected_action_size + 1
        mat = np.zeros((len(self._queue), entry_size))
        for i, (s, a, r) in enumerate(self._queue):
            vec = np.concatenate([to_1d_array(s), to_1d_array(a), [r]])
            if vec.size < entry_size:
                vec = np.concatenate([vec, np.zeros(entry_size - vec.size)])
            elif vec.size > entry_size:
                vec = vec[:entry_size]
            mat[i] = vec
        if self._use_differences:
            if len(self._queue) < 2:
                return np.zeros(self._calculator._num_state)
            diffs = np.diff(mat, axis=0)
            big_state = diffs.flatten()
            if diffs.shape[0] < self._num_ar - 1:
                pad = np.zeros((self._num_ar - 1 - diffs.shape[0]) * entry_size)
                big_state = np.concatenate([big_state, pad])
            return big_state
        else:
            big_state = mat.flatten()
            if len(self._queue) < self._num_ar:
                pad = np.zeros((self._num_ar - len(self._queue)) * entry_size)
                big_state = np.concatenate([big_state, pad])
            return big_state

    def big_call(self, state, action, reward):
        self._queue.append((state, action, reward))
        big_state = self._create_big_state()
        return self._calculator.calculate(big_state)
