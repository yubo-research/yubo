from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from sampling.log_uniform import np_log_uniform


@dataclass
class TYStaggerTR:
    num_dim: int
    _num_arms: int
    s_min: float = 1e-4
    length_sampler: Optional[Callable[[float, float], float]] = None

    def update_from_model(self, Y):
        pass

    def pre_draw(self):
        pass

    def create_trust_region(self, x_center, kernel):
        if hasattr(kernel, "lengthscale"):
            weights = kernel.lengthscale.cpu().detach().numpy().ravel()
            weights = weights / weights.mean()
            weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))
        else:
            weights = np.ones(self.num_dim)
        sampler = self.length_sampler or np_log_uniform
        length = sampler(self.s_min, 1.0)
        lb = np.clip(x_center.cpu().numpy() - weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center.cpu().numpy() + weights * length / 2.0, 0.0, 1.0)
        return lb, ub
