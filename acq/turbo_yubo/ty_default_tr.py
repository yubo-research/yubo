from dataclasses import dataclass

import numpy as np


class TurboYUBORestartError(Exception):
    pass


@dataclass
class TYDefaultTR:
    num_dim: int

    _num_arms: int
    _length: float = 0.8
    _length_init: float = 0.8
    _length_min: float = 0.5**7
    _length_max: float = 1.6
    _failure_counter: int = 0
    _failure_tolerance: int = float("nan")
    _success_counter: int = 0
    _success_tolerance: int = 3
    _best_value: float = -float("inf")
    _restart_triggered: bool = False
    _prev_y_length: int = 0

    def __post_init__(self):
        self._failure_tolerance = np.ceil(max([4.0 / self._num_arms, float(self.num_dim) / self._num_arms]))

    def update_from_model(self, Y):
        if len(Y) > self._prev_y_length:
            new_Y = Y[self._prev_y_length :]
            self._update_state(new_Y)
            self._prev_y_length = len(Y)

    def pre_draw(self):
        if self._restart_triggered:
            self._restart()

    def create_trust_region(self, x_center, kernel):
        if hasattr(kernel, "lengthscale"):
            weights = kernel.lengthscale.cpu().detach().numpy().ravel()
            weights = weights / weights.mean()
            weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))
        else:
            weights = np.ones(self.num_dim)
        lb = np.clip(x_center.cpu().numpy() - weights * self._length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center.cpu().numpy() + weights * self._length / 2.0, 0.0, 1.0)
        return lb, ub

    def _update_state(self, Y_next):
        if len(Y_next) == 0:
            return self
        if not np.isfinite(self._best_value):
            self._best_value = max(Y_next).item()
            self._prev_y_length += len(Y_next)
            return self
        if max(Y_next) > self._best_value + 1e-3 * np.fabs(self._best_value):
            self._success_counter += 1
            self._failure_counter = 0
        else:
            self._success_counter = 0
            self._failure_counter += 1

        if self._success_counter == self._success_tolerance:
            self._length = min(2.0 * self._length, self._length_max)
            self._success_counter = 0
        elif self._failure_counter == self._failure_tolerance:
            self._length /= 2.0
            self._failure_counter = 0

        self._best_value = max(self._best_value, max(Y_next).item())
        if self._length < self._length_min:
            self._restart_triggered = True
        return self

    def _restart(self):
        self._length = self._length_init
        self._success_counter = 0
        self._failure_counter = 0
        self._best_value: float = -float("inf")
        self._prev_y_length = 0
        self._restart_triggered = False
        raise TurboYUBORestartError()
