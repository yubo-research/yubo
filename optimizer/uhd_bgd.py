import math

from .gaussian_perturbator import GaussianPerturbator
from .step_size_adapter import StepSizeAdapter


class UHDBGD:
    """Bandit gradient ascent.

    Uses baselined change in mu as the gradient signal:
      signal = (mu - mu_prev) - alpha * EWMA(mu - mu_prev)
    This removes the expected drift, leaving only the surprise
    attributable to the perturbation direction.
    """

    def __init__(
        self,
        perturbator: GaussianPerturbator,
        sigma_0: float,
        dim: int,
        *,
        lr: float = 0.001,
        alpha: float = 0.1,
        beta: float = 0.9,
    ):
        self._perturbator = perturbator
        self._adapter = StepSizeAdapter(sigma_0=sigma_0, dim=dim)
        self._seed = 0
        self._lr = lr
        self._alpha = alpha
        self._beta = beta
        self._sqrt_dim = math.sqrt(dim)
        self._y_best: float | None = None
        self._mu_prev: float | None = None
        self._se_prev = 0.0
        self._ewma_delta = 0.0

    @property
    def sigma(self) -> float:
        return self._adapter.sigma

    @property
    def y_best(self) -> float | None:
        return self._y_best

    @property
    def mu_avg(self) -> float:
        return self._mu_prev if self._mu_prev is not None else 0.0

    @property
    def se_avg(self) -> float:
        return self._se_prev

    def ask(self) -> None:
        self._perturbator.perturb(self._seed, self._adapter.sigma)

    def tell(self, mu: float, se: float) -> None:
        self._se_prev = se

        if self._y_best is None or mu > self._y_best:
            self._y_best = mu

        # Unperturb to restore x.
        self._perturbator.unperturb()

        if self._mu_prev is None:
            # First call: no previous mu to compare against. No step.
            self._mu_prev = mu
            self._perturbator.perturb(self._seed, 0.0)
            self._perturbator.accept()
            self._seed += 1
            return

        # Baselined change: subtract expected drift.
        delta = mu - self._mu_prev
        signal = delta - self._alpha * self._ewma_delta
        self._ewma_delta += (1 - self._beta) * (delta - self._ewma_delta)
        self._mu_prev = mu

        # Gradient step: params += lr * (signal / sigma) * epsilon / sqrt(dim).
        sigma = self._adapter.sigma
        step_scale = self._lr * signal / (sigma * self._sqrt_dim)
        self._perturbator.perturb(self._seed, step_scale)
        self._perturbator.accept()

        self._seed += 1
