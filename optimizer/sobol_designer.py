import numpy as np
from scipy.stats import qmc

import common.all_bounds as all_bounds


class SobolDesigner:
    def __init__(self, policy, init_center, max_points=2**10):
        self._policy = policy
        seed = np.random.randint(99999)
        self._ps = qmc.Sobol(policy.num_params(), seed=seed).random(max_points)
        self._init_center = init_center

    def init_center(self):
        return self._init_center

    def __call__(self, data, num_arms):
        assert len(self._ps) > 0, "max_points exceeded"

        policies = []
        for _ in range(num_arms):
            p = self._ps[0, :]
            self._ps = self._ps[1:, :]
            policy = self._policy.clone()
            policy.set_params(all_bounds.p_low + all_bounds.p_width * p)
            policies.append(policy)
        return policies
