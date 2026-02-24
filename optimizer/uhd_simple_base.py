from __future__ import annotations

import numpy as np

from .gaussian_perturbator import GaussianPerturbator
from .step_size_adapter import StepSizeAdapter


class UHDSimpleBase:
    """Shared state and logic for (1+1)-ES optimizers."""

    def __init__(
        self,
        perturbator: GaussianPerturbator,
        sigma_0: float,
        dim: int,
        *,
        sigma_range: tuple[float, float] | None = None,
    ):
        self._perturbator = perturbator
        self._adapter = StepSizeAdapter(sigma_0=sigma_0, dim=dim)
        self._sigma_range = sigma_range
        self._eval_seed = 0
        self._y_best: float | None = None
        self._mu_prev = 0.0
        self._se_prev = 0.0

    @property
    def eval_seed(self) -> int:
        return self._eval_seed

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

    def _accept_or_reject(self, mu: float) -> None:
        if self._y_best is None or mu > self._y_best:
            self._y_best = mu
            self._adapter.update(accepted=True)
            self._perturbator.accept()
        else:
            self._adapter.update(accepted=False)
            self._perturbator.unperturb()

    def _sample_sigmas(self, base_seed: int, n: int) -> np.ndarray:
        if self._sigma_range is None:
            return np.full(n, self._adapter.sigma)
        lo, hi = np.log(self._sigma_range[0]), np.log(self._sigma_range[1])
        rng = np.random.default_rng(base_seed)
        return np.exp(rng.uniform(lo, hi, size=n))
