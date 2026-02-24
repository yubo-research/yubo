from __future__ import annotations

import numpy as np

from .step_size_adapter import StepSizeAdapter


class UHDSimpleNp:
    """(1+1)-ES for numpy-based policies (no nn.Module required)."""

    def __init__(self, policy, *, sigma_0: float, param_clip: tuple[float, float] | None = None):
        self._policy = policy
        self._x = np.asarray(policy.get_params(), dtype=np.float64).copy()
        dim = len(self._x)
        self._adapter = StepSizeAdapter(sigma_0=sigma_0, dim=dim)
        self._param_clip = param_clip
        self._seed = 0
        self._y_best: float | None = None
        self._mu_prev = 0.0
        self._se_prev = 0.0
        self._x_candidate: np.ndarray | None = None

    @property
    def eval_seed(self) -> int:
        return self._seed

    @property
    def sigma(self) -> float:
        return self._adapter.sigma

    @property
    def y_best(self) -> float | None:
        return self._y_best

    @property
    def mu_avg(self) -> float:
        return self._mu_prev

    @property
    def se_avg(self) -> float:
        return self._se_prev

    def _clip(self, x: np.ndarray) -> np.ndarray:
        if self._param_clip is not None:
            return np.clip(x, self._param_clip[0], self._param_clip[1])
        return x

    def ask(self) -> None:
        rng = np.random.default_rng(self._seed)
        noise = rng.standard_normal(len(self._x))
        self._x_candidate = self._clip(self._x + self._adapter.sigma * noise)
        self._policy.set_params(self._x_candidate)

    def tell(self, mu: float, se: float) -> None:
        self._mu_prev = mu
        self._se_prev = se

        if self._y_best is None or mu > self._y_best:
            self._y_best = mu
            self._adapter.update(accepted=True)
            self._x = self._x_candidate.copy()
        else:
            self._adapter.update(accepted=False)
            self._policy.set_params(self._x)

        self._seed += 1
