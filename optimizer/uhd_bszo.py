from .gaussian_perturbator import GaussianPerturbator, apply_weight_decay
from .lr_scheduler import LRScheduler


class _KalmanFilter:
    def __init__(self, k: int, sigma_p_sq: float, sigma_e_sq: float, alpha: float):
        self._k = k
        self._sigma_p_sq = sigma_p_sq
        self.sigma_e_sq = sigma_e_sq
        self._alpha = alpha
        self.mu_post: list[float] = [0.0] * k
        self.Sigma: list[list[float]] = _zero_matrix(k)
        self.Y: list[float] = [0.0] * k
        self.last_d_idx = 0
        self.last_y = 0.0

    def init_step(self) -> None:
        k = self._k
        self.mu_post = [0.0] * k
        self.Sigma = [[self._sigma_p_sq if i == j else 0.0 for j in range(k)] for i in range(k)]

    def update_coord(self, idx: int, y: float) -> None:
        k = self._k
        S = self.Sigma
        mu = self.mu_post

        innov = y - mu[idx]
        denom = S[idx][idx] + self.sigma_e_sq
        if abs(denom) < 1e-30:
            return
        K = [S[i][idx] / denom for i in range(k)]

        for i in range(k):
            mu[i] += K[i] * innov

        row_idx = [S[idx][j] for j in range(k)]
        for i in range(k):
            for j in range(k):
                S[i][j] -= K[i] * row_idx[j]

    def adaptive_step(self) -> None:
        r = self.last_y - self.mu_post[self.last_d_idx]
        self.sigma_e_sq = (1.0 - self._alpha) * self.sigma_e_sq + self._alpha * r * r

        j = max(range(self._k), key=lambda i: self.Sigma[i][i])
        self.update_coord(j, self.Y[j])


class UHDBSZO:
    def __init__(
        self,
        perturbator: GaussianPerturbator,
        dim: int,
        *,
        lr_scheduler: LRScheduler,
        epsilon: float = 1e-4,
        k: int = 2,
        sigma_p_sq: float = 1.0,
        sigma_e_sq: float = 1.0,
        alpha: float = 0.1,
        weight_decay: float = 0.0,
    ):
        self._perturbator = perturbator
        self._dim = dim
        self._lr_scheduler = lr_scheduler
        self._epsilon = epsilon
        self._k = k
        self._weight_decay = weight_decay

        self._kf = _KalmanFilter(k, sigma_p_sq, sigma_e_sq, alpha)

        self._eval_seed = 0
        self._perturb_base = 0
        self._phase = 0
        self._f0 = 0.0

        self._y_best: float | None = None
        self._mu_prev = 0.0
        self._se_prev = 0.0

    @property
    def eval_seed(self) -> int:
        return self._eval_seed

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def y_best(self) -> float | None:
        return self._y_best

    @property
    def mu_avg(self) -> float:
        return self._mu_prev

    @property
    def se_avg(self) -> float:
        return self._se_prev

    @property
    def phase(self) -> int:
        return self._phase

    @property
    def k(self) -> int:
        return self._k

    @property
    def perturbator(self) -> GaussianPerturbator:
        return self._perturbator

    @property
    def baseline_mu(self) -> float:
        return self._f0

    def perturb_seed(self, direction: int) -> int:
        return self._perturb_base + direction

    def set_perturb_base(self, base: int) -> None:
        if self._phase != 0:
            raise RuntimeError("set_perturb_base only valid during baseline phase")
        self._perturb_base = int(base)

    def ask(self) -> None:
        if self._phase == 0:
            return
        i = self._phase - 1
        self._perturbator.perturb(self.perturb_seed(i), self._epsilon)

    def tell(self, mu: float, se: float) -> None:
        self._mu_prev = mu
        self._se_prev = se
        if self._y_best is None or mu > self._y_best:
            self._y_best = mu

        if self._phase == 0:
            self._f0 = mu
            self._kf.init_step()
            self._phase = 1
            return

        kf = self._kf
        i = self._phase - 1
        y_i = (mu - self._f0) / self._epsilon
        kf.Y[i] = y_i
        self._perturbator.unperturb()

        kf.update_coord(i, y_i)
        kf.last_d_idx = i
        kf.last_y = y_i

        if self._phase == self._k:
            kf.adaptive_step()
            self._apply_gradient()
            self._lr_scheduler.step()
            self._eval_seed += 1
            self._perturb_base += self._k
            self._phase = 0
        else:
            self._phase += 1

    def _apply_gradient(self) -> None:
        lr = self._lr_scheduler.lr
        for i in range(self._k):
            scale = lr * self._kf.mu_post[i]
            if abs(scale) < 1e-30:
                continue
            self._perturbator.perturb(self.perturb_seed(i), scale)
            self._perturbator.accept()

        apply_weight_decay(self._perturbator._module, lr, self._weight_decay)


def _zero_matrix(k: int) -> list[list[float]]:
    return [[0.0] * k for _ in range(k)]
