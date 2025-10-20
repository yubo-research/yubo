from dataclasses import dataclass

from acq.turbo_yubo.ty_default_tr import mk_lb_ub_from_kernel
from sampling.log_uniform import np_log_uniform


@dataclass
class TYStaggerTR:
    num_dim: int
    num_arms: int

    s_min: float = 1e-4
    s_max: float = 1.0

    def update_from_model(self, Y):
        pass

    def pre_draw(self):
        pass

    def create_trust_region(self, x_center, kernel):
        length = np_log_uniform(self.s_min, self.s_max)
        return mk_lb_ub_from_kernel(x_center, kernel, length)
