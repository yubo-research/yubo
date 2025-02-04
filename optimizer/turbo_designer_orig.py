import numpy as np

import common.all_bounds as all_bounds
from turbo_m_ref.turbo_1_ask_tell import Turbo1
from turbo_m_ref.turbo_m_ask_tell import TurboM


class TuRBODesigner:
    def __init__(self, policy, num_init=None):
        self._policy = policy
        self._num_init = num_init
        self._bounds = np.array([[all_bounds.x_low, all_bounds.x_high]] * policy.num_params())
        self._turbo = None
        self._num_arms = None

    def __call__(self, data, num_arms):
        if self._num_arms is None:
            self._num_arms = num_arms
            num_init = self._num_init if self._num_init else num_arms
            if num_arms == 1:
                self._turbo = Turbo1(self._bounds, n_init=num_init)
            else:
                self._turbo = TurboM(self._bounds, n_init=num_init)

        if len(data) > 0:
            y = data[-1].trajectory.rreturn
            x = data[-1].policy.get_params()
            self._turbo_1.tell(y, x)

        p = self._turbo_1.ask()
        policy = self._policy.clone()
        policy.set_params(p)
        return [policy]
