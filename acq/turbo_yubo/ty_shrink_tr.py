from dataclasses import dataclass

from acq.turbo_yubo.ty_default_tr import mk_lb_ub_from_kernel


@dataclass
class TYShrinkTR:
    num_dim: int
    num_arms: int

    s_min: float = 0.5**7
    s_max: float = 1.0

    def update_from_model(self, Y):
        pass

    def pre_draw(self):
        pass

    def create_trust_region(self, x_center, kernel, num_obs):
        length = num_obs**self.num_dim
        return mk_lb_ub_from_kernel(x_center, kernel, length)
