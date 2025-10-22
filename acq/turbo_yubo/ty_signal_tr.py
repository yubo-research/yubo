from dataclasses import dataclass

import numpy as np

from acq.turbo_yubo.ty_default_tr import mk_lb_ub_from_kernel
from sampling.sampling_util import gumbel


def ty_signal_tr_factory_factory(use_gumbel: bool = False):
    def _factory(*, num_dim: int, num_arms: int):
        return TYSignalTR(num_dim=num_dim, num_arms=num_arms, use_gumbel=use_gumbel)

    return _factory


@dataclass
class TYSignalTR:
    num_dim: int
    num_arms: int
    use_gumbel: bool = False

    s_min: float = 0.1  # 0.5**7
    s_max: float = 1.0

    _signal: float | None = None

    def update_from_model(self, y):
        y = y.detach().cpu().numpy()
        if len(y) <= 1:
            self._signal = 0
        else:
            denom = 2 * gumbel(len(y)) if self.use_gumbel else 4.33
            self._signal = ((y.max() - np.median(y)) / (1e-6 + y.std()) / denom) ** 2

    def pre_draw(self):
        pass

    def create_trust_region(self, x_center, kernel, num_obs):
        assert self._signal is not None
        scale = 1.0 / (1e-6 + self._signal)
        # scale = scale**self.num_dim
        length = np.minimum(self.s_max, np.maximum(self.s_min, scale))
        print("length:", length, self._signal)
        return mk_lb_ub_from_kernel(x_center, kernel, length)
