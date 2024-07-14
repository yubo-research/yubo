import numpy as np
from scipy.stats import qmc

import common.all_bounds as all_bounds


class SobolDesigner:
    def __init__(self, policy, init_center, max_points=2**12):
        self._policy = policy
        seed = policy.problem_seed + 12345
        self._max_points = int(2 ** np.ceil(np.log2(max_points)))
        self._init_center = init_center
        self.seed = seed
        self._i_seed = 0
        self._ps = []
        self._reset()

    def init_center(self):
        return self._init_center

    def estimate(self, data, X):
        return [None] * len(X)

    def _reset(self):
        if len(self._ps) == 0:
            self._ps = qmc.Sobol(self._policy.num_params(), seed=self.seed + self._i_seed).random(self._max_points)
            self._i_seed += 1

    def __call__(self, _, num_arms):
        policies = []
        self.fig_last_arms = []
        for _ in range(num_arms):
            self._reset()
            x = self._ps[0, :]
            self.fig_last_arms.append(x)
            self._ps = self._ps[1:, :]
            policy = self._policy.clone()
            policy.set_params(all_bounds.p_low + all_bounds.p_width * x)
            policies.append(policy)
        return policies
