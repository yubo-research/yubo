import numpy as np

from .gaussian_perturbator import GaussianPerturbator
from .step_size_adapter import StepSizeAdapter

_SE_FLOOR = 1e-10


class UHDHoeffding:
    def __init__(
        self,
        perturbator: GaussianPerturbator,
        sigma_0: float,
        dim: int,
        *,
        alpha: float = 0.0,
    ):
        self._perturbator = perturbator
        self._adapter = StepSizeAdapter(sigma_0=sigma_0, dim=dim)

        self._sum_w_best: float | None = None
        self._sum_w_mu_best: float | None = None
        self._seed_best: int | None = None

        self._sum_w = 0.0
        self._sum_w_mu = 0.0
        self._seed = 0

        self._alpha = alpha

    @property
    def eval_seed(self) -> int:
        return self._seed

    @property
    def sigma(self) -> float:
        return self._adapter.sigma

    @property
    def y_best(self) -> float | None:
        if self._sum_w_best is None:
            return None
        return self._sum_w_mu_best / self._sum_w_best

    @property
    def mu_avg(self) -> float:
        return self._mu_prev

    @property
    def se_avg(self) -> float:
        return self._se_prev

    def ask(self) -> None:
        self._perturbator.perturb(self._seed, self._adapter.sigma)

    def tell(self, mu: float, se: float) -> None:
        w = 1.0 / max(se, _SE_FLOOR) ** 2
        self._sum_w += w
        self._sum_w_mu += w * mu
        mu_curr = self._sum_w_mu / self._sum_w
        se_curr = 1.0 / np.sqrt(self._sum_w)

        self._mu_prev = mu_curr
        self._se_prev = se_curr

        if self._sum_w_best is None:
            mu_best, se_best = None, None
        else:
            mu_best = self._sum_w_mu_best / self._sum_w_best
            se_best = 1.0 / np.sqrt(self._sum_w_best)

        if mu_best is None or self._better(mu_curr, se_curr, mu_best, se_best):
            self._sum_w_best = self._sum_w
            self._sum_w_mu_best = self._sum_w_mu
            self._seed_best = self._seed
            self._adapter.update(accepted=True)
            self._perturbator.accept()
            self._seed += 1
        elif self._worse(mu_curr, se_curr, mu_best, se_best):
            self._adapter.update(accepted=False)
            self._perturbator.unperturb()
            self._seed += 1
            self._reset()
        elif se_curr < se_best:
            self._perturbator.unperturb()
            self._seed, self._seed_best = self._seed_best, self._seed
            self._sum_w_best, self._sum_w = self._sum_w, self._sum_w_best
            self._sum_w_mu_best, self._sum_w_mu = self._sum_w_mu, self._sum_w_mu_best
        else:
            self._perturbator.unperturb()

    def _reset(self) -> None:
        self._sum_w = 0.0
        self._sum_w_mu = 0.0

    def _inside(self, mu_a: float, se_a: float, mu_b: float, se_b: float) -> bool:
        return (mu_a + self._alpha * se_a > mu_b + self._alpha * se_b) and (mu_a - self._alpha * se_a < mu_b - self._alpha * se_b)

    def _better(self, mu_a: float, se_a: float, mu_b: float, se_b: float) -> bool:
        return mu_a - self._alpha * se_a > mu_b + self._alpha * se_b

    def _worse(self, mu_a: float, se_a: float, mu_b: float, se_b: float) -> bool:
        return mu_a + self._alpha * se_a < mu_b - self._alpha * se_b
