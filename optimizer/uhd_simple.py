from .gaussian_perturbator import GaussianPerturbator
from .step_size_adapter import StepSizeAdapter


class UHDSimple:
    """Simplest (1+1)-ES: accept if mu improves, reject otherwise."""

    def __init__(
        self,
        perturbator: GaussianPerturbator,
        sigma_0: float,
        dim: int,
    ):
        self._perturbator = perturbator
        self._adapter = StepSizeAdapter(sigma_0=sigma_0, dim=dim)
        self._seed = 0
        self._y_best: float | None = None
        self._mu_prev = 0.0
        self._se_prev = 0.0

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

    def ask(self) -> None:
        self._perturbator.perturb(self._seed, self._adapter.sigma)

    def tell(self, mu: float, se: float) -> None:
        self._mu_prev = mu
        self._se_prev = se

        if self._y_best is None or mu > self._y_best:
            self._y_best = mu
            self._adapter.update(accepted=True)
            self._perturbator.accept()
        else:
            self._adapter.update(accepted=False)
            self._perturbator.unperturb()

        self._seed += 1
