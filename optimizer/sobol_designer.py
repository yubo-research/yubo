import numpy as np
from scipy.stats import qmc

import common.all_bounds as all_bounds


class SobolDesigner:
    def __init__(self, policy, init_center, max_points=2**12):
        self._policy = policy
        seed = policy.seed + 12345
        max_points = int(2 ** np.ceil(np.log2(max_points)))
        self._ps = qmc.Sobol(policy.num_params(), seed=seed).random(max_points)
        self._init_center = init_center
        self.seed = seed

    def init_center(self):
        return self._init_center

    def __call__(self, _, num_arms):
        assert len(self._ps) > 0, "max_points exceeded"

        policies = []
        self.fig_last_arms = []
        for _ in range(num_arms):
            x = self._ps[0, :]
            self.fig_last_arms.append(x)
            self._ps = self._ps[1:, :]
            policy = self._policy.clone()
            policy.set_params(all_bounds.p_low + all_bounds.p_width * x)
            policies.append(policy)
        return policies
