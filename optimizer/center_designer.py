import numpy as np

import common.all_bounds as all_bounds


class CenterDesigner:
    def __init__(self, policy):
        self._policy = policy

    def __call__(self, data, num_arms, *, telemetry=None):
        assert num_arms == 1, num_arms
        policy = self._policy.clone()
        p = (all_bounds.p_low + 0.5 * all_bounds.p_width) * np.ones(
            shape=(policy.num_params(),)
        )

        policy.set_params(p)
        return [policy]
