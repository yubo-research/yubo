from scipy.stats import qmc

import common.all_bounds as all_bounds


class SobolDesigner:
    def __init__(self, policy, init_center):
        self._policy = policy
        max_points = 10240  # TODO: worry later
        self._ps = qmc.Sobol(policy.num_params()).random(max_points)
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
