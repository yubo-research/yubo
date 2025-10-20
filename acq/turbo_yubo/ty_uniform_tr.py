from dataclasses import dataclass

import numpy as np

from acq.turbo_yubo.ty_default_tr import mk_lb_ub_from_kernel


@dataclass
class TYUniformTR:
    num_dim: int
    num_arms: int

    s_min: float = 0.5**7
    s_max: float = 1.0

    def update_from_model(self, Y):
        pass

    def pre_draw(self):
        pass

    def create_trust_region(self, x_center, kernel, num_obs):
        length = np.random.uniform(self.s_min, self.s_max)
        print("LENGTH:", length)
        return mk_lb_ub_from_kernel(x_center, kernel, length)
