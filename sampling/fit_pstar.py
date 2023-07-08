import numpy as np

from .util import draw_bounded_normal_samples


class FitPStar:
    def __init__(self, mu, cov_aspect, alpha=0.5, exp_low_0=-20):
        self._mu = mu
        self._num_dim = len(mu)
        assert len(mu) == len(cov_aspect), (
            "cov_aspect should be a vector, representing a diagonal covariance matrix, and it should have the same length as mu",
            len(mu),
            len(cov_aspect),
        )
        cov_aspect = np.abs(cov_aspect)
        cov_aspect = cov_aspect / cov_aspect.mean()
        det = np.prod(cov_aspect)
        self.unit_cov_diag = cov_aspect / (det ** (1 / self._num_dim))
        self._scale2_exp_low = exp_low_0
        self._scale2_exp_high = 0
        self._alpha = 0.5

    def _mid(self):
        return (self._scale2_exp_low + self._scale2_exp_high) / 2

    def converged(self):
        return (self._scale2_exp_high - self._scale2_exp_low) < 1

    def scale2(self):
        return np.exp(self._mid())

    def cov(self):
        return self.scale2() * self.unit_cov_diag

    def sigma(self):
        return np.sqrt(self.scale2())

    def mu(self):
        return self._mu

    def ask(self, num_samples, qmc=False):
        return draw_bounded_normal_samples(self._mu, self.cov(), num_samples, qmc=qmc)

    def tell(self, x, pi):
        x = np.asarray(x)
        pi = np.asarray(pi)

        dx = x - self._mu
        w = 1 / pi

        d2 = (w[:, None] * dx * dx).sum(axis=0) / w.sum()
        scale2_est = d2.mean()

        if scale2_est > self.scale2():
            self._scale2_exp_low = (1 - self._alpha) * self._scale2_exp_low + self._alpha * self._mid()
        else:
            self._scale2_exp_high = (1 - self._alpha) * self._scale2_exp_high + self._alpha * self._mid()
