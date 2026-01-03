import time

import numpy as np

import common.all_bounds as all_bounds


class RandomDesigner:
    def __init__(self, policy):
        self._policy = policy

    def __call__(self, data, num_arms, telemetry=None):
        t0 = time.time()
        policies = []
        for _ in range(num_arms):
            policy = self._policy.clone()
            p = policy.get_params()
            p = np.random.uniform(all_bounds.p_low, all_bounds.p_high, size=p.shape)
            policy.set_params(p)
            policies.append(policy)
        dt_sel = time.time() - t0
        if telemetry is not None:
            telemetry.set_dt_fit(0)
            telemetry.set_dt_select(dt_sel)
        return policies
