import numpy as np

from .util import mk_normal_samples


class PStar:
    def __init__(self, mu, cov_aspect, sigma_0, alpha=1.0):
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
        self._unit_cov_diag = cov_aspect / (det ** (1 / self._num_dim))
        self._num = 0
        self._scale2 = sigma_0**2
        self._alpha = alpha

    def sigma(self):
        return np.sqrt(self._scale2)

    def ask(self, num_samples):
        cov = (self.sigma() ** 2) * self._unit_cov_diag
        return mk_normal_samples([(self._mu, cov)], num_samples)

    def _mk_x_pi(self, samples):
        x = []
        pi = []
        for s in samples:
            x.append(s.x)
            pi.append(s.p)

        x = np.array(x)
        pi = np.array(pi)
        return x, pi

    def tell(self, resamples):
        x, pi = self._mk_x_pi(resamples)

        dx = x - self._mu
        w = 1 / pi

        d2 = (w[:, None] * dx * dx).sum(axis=0) / w.sum()
        scale2_est = d2.mean()

        self._num += self._alpha * (len(pi) - self._num)
        self._scale2 += self._alpha * (scale2_est - self._scale2)
