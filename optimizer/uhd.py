from .gaussian_perturbator import GaussianPerturbator
from .step_size_adapter import StepSizeAdapter


class UHD:
    def __init__(self, perturbator: GaussianPerturbator, sigma_0: float, dim: int):
        self._perturbator = perturbator
        self._adapter = StepSizeAdapter(sigma_0=sigma_0, dim=dim)
        self._seed = 0
        self._y_max: float | None = None

    @property
    def y_max(self) -> float | None:
        return self._y_max

    @property
    def sigma(self) -> float:
        return self._adapter.sigma

    def ask(self) -> int:
        seed = self._seed
        self._seed += 1
        self._perturbator.perturb(seed, self._adapter.sigma)
        return seed

    def tell(self, seed: int, y: float) -> None:
        improved = self._adapter.update(y)
        if improved:
            self._y_max = y
            self._perturbator.accept()
        else:
            self._perturbator.unperturb()
