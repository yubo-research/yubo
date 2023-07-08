import numpy as np

from .util import draw_bounded_normal_samples


class FitPStar:
    def __init__(self, mu, cov_aspect, sigma_0=0.3, alpha=1.0):
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
        self._scale2 = sigma_0**2
        self._alpha = alpha

    def scale2(self):
        return self._scale2

    def cov(self):
        return self._scale2 * self.unit_cov_diag

    def sigma(self):
        return np.sqrt(self._scale2)

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

        self._scale2 += self._alpha * (scale2_est - self._scale2)
