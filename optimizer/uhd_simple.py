from __future__ import annotations

from .gaussian_perturbator import GaussianPerturbator
from .uhd_simple_base import UHDSimpleBase


class UHDSimple(UHDSimpleBase):
    """Simplest (1+1)-ES: accept if mu improves, reject otherwise."""

    def __init__(
        self,
        perturbator: GaussianPerturbator,
        sigma_0: float,
        dim: int,
        *,
        sigma_range: tuple[float, float] | None = None,
    ):
        super().__init__(perturbator, sigma_0, dim, sigma_range=sigma_range)

    def ask(self) -> None:
        sigma = float(self._sample_sigmas(self._eval_seed, 1)[0])
        self._perturbator.perturb(self._eval_seed, sigma)

    def tell(self, mu: float, se: float) -> None:
        self._mu_prev = mu
        self._se_prev = se
        self._accept_or_reject(mu)
        self._eval_seed += 1
