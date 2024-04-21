import numpy as np

import common.all_bounds as all_bounds
from turbo_m_ref.turbo_1_ask_tell import Turbo1


class TuRBODesigner:
    def __init__(self, policy, num_init=None):
        self._policy = policy
        bounds = np.array([[all_bounds.x_low, all_bounds.x_high]] * policy.num_params())
        self._turbo = Turbo1(bounds, n_init=num_init)

    def __call__(self, data, num_arms):
        assert num_arms == 1, ("NYI (but possible), num_arms>1", num_arms)

        if len(data) > 0:
            y = data[-1].trajectory.rreturn
            x = data[-1].policy.get_params()
            self._turbo.tell(y, x)

        p = self._turbo.ask()
        policy = self._policy.clone()
        policy.set_params(p)
        return [policy]
