import numpy as np

import common.all_bounds as all_bounds


class RandomDesigner:
    def __init__(self, policy):
        self._policy = policy

<<<<<<< HEAD
    def __call__(self, data):
        policy = self._policy.clone()
        p = policy.get_params()
        p = np.random.uniform(all_bounds.p_low, all_bounds.p_high, size=p.shape)
        policy.set_params(p)
        return policy
=======
    def __call__(self, data, num_arms):
        policies = []
        for _ in range(num_arms):
            policy = self._policy.clone()
            p = policy.get_params()
            p = np.random.uniform(all_bounds.p_low, all_bounds.p_high, size=p.shape)
            policy.set_params(p)
            policies.append(policy)
        return policies
>>>>>>> main
