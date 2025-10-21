from dataclasses import dataclass

import numpy as np

from acq.turbo_yubo.ty_default_tr import mk_lb_ub_from_kernel


@dataclass
class TYSignalTR:
    num_dim: int
    num_arms: int

    s_min: float = 0.5**7
    s_max: float = 1.0

    _signal: float | None = None

    def update_from_model(self, y):
        y = y.detach().cpu().numpy()
        if len(y) == 0:
            self._signal = 0
        else:
            self._signal = (y.max() - np.median(y)) / (1e-6 + y.std()) / 4.33

    def pre_draw(self):
        pass

    def create_trust_region(self, x_center, kernel, num_obs):
        assert self._signal is not None
        length = np.minimum(self.s_max, np.maximum(self.s_min, 0.1 / (1e-6 + self._signal)))
        print("length:", length, self._signal)
        return mk_lb_ub_from_kernel(x_center, kernel, length)
