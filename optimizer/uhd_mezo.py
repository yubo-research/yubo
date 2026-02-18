import math

from .gaussian_perturbator import GaussianPerturbator
from .lr_scheduler import LRScheduler
from .step_size_adapter import StepSizeAdapter


class UHDMeZO:
    """Gradient ascent via antithetic sampling.

    Each gradient step uses two sequential ask/tell cycles:
      ask  1: perturb x → x + σε       (positive direction)
      tell 1: record mu_plus
      ask  2: perturb x → x − σε       (negative direction)
      tell 2: apply gradient step  lr · g_norm · ε

    where g_norm = projected_grad / rms(projected_grad) (Adam-style).
    No dimension-sized storage: noise is replayed from the seed.
    """

    def __init__(
        self,
        perturbator: GaussianPerturbator,
        dim: int,
        *,
        lr_scheduler: LRScheduler,
        sigma: float = 0.001,
        beta: float = 0.9,
        weight_decay: float = 0.0,
    ):
        self._perturbator = perturbator
        self._adapter = StepSizeAdapter(dim=dim, sigma_0=sigma, sigma_min=sigma, sigma_max=sigma)
        self._lr_scheduler = lr_scheduler
        self._beta = beta
        self._weight_decay = weight_decay
        self._grad_sq_ema = 0.0
        self._seed = 0
        self._y_best: float | None = None
        self._mu_prev = 0.0
        self._se_prev = 0.0

        # Alternates between positive and negative phase.
        self._positive_phase = True
        self._mu_plus = 0.0
        self._step_seed = 0
        self._step_sigma = 0.0
        self._last_step_scale = 0.0

    @property
    def eval_seed(self) -> int:
        return self._seed

    def set_next_seed(self, seed: int) -> None:
        """Override the next seed used for the upcoming positive phase.

        This is useful for seed-filtering strategies (e.g. ENN-based candidate selection).
        Only valid at the start of a new antithetic pair (positive phase).
        """
        if not self._positive_phase:
            raise RuntimeError("set_next_seed is only valid during positive phase")
        self._seed = int(seed)

    def skip_negative(self) -> None:
        """Skip the negative phase for the current pair and advance to the next seed.

        Intended for early-reject strategies: after observing mu_plus, decide to not
        spend the x- evaluation and also not apply an update.
        """
        if self._positive_phase:
            raise RuntimeError("skip_negative is only valid after the positive phase")
        # We're currently at baseline params (positive tell unperturbed). Move on.
        self._positive_phase = True
        self._seed += 1

    @property
    def positive_phase(self) -> bool:
        return self._positive_phase

    @property
    def step_seed(self) -> int:
        return self._step_seed

    @property
    def step_sigma(self) -> float:
        return self._step_sigma

    @property
    def last_step_scale(self) -> float:
        return self._last_step_scale

    @property
    def perturbator(self) -> GaussianPerturbator:
        return self._perturbator

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
        if self._positive_phase:
            self._step_sigma = self._adapter.sigma
            self._step_seed = self._seed
            self._perturbator.perturb(self._seed, self._step_sigma)
        else:
            self._perturbator.perturb(self._step_seed, -self._step_sigma)

    def tell(self, mu: float, se: float) -> None:
        self._mu_prev = mu
        self._se_prev = se

        if self._y_best is None or mu > self._y_best:
            self._y_best = mu

        if self._positive_phase:
            self._mu_plus = mu
            self._perturbator.unperturb()
            self._positive_phase = False
        else:
            mu_minus = mu
            self._perturbator.unperturb()

            projected_grad = (self._mu_plus - mu_minus) / (2.0 * self._step_sigma)
            self._grad_sq_ema = self._beta * self._grad_sq_ema + (1.0 - self._beta) * projected_grad**2
            rms = math.sqrt(self._grad_sq_ema) + 1e-8
            step_scale = self._lr_scheduler.lr * projected_grad / rms
            self._last_step_scale = float(step_scale)

            self._perturbator.perturb(self._step_seed, step_scale)
            self._perturbator.accept()

            if self._weight_decay > 0.0:
                decay = 1.0 - self._lr_scheduler.lr * self._weight_decay
                for p in self._perturbator._module.parameters():
                    p.data.mul_(decay)

            self._lr_scheduler.step()
            self._seed += 1
            self._positive_phase = True
