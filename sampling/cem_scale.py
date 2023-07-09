import numpy as np

from .scaled_inv_chi2 import ScaledInverseChi2
from .util import draw_bounded_normal_samples


class CEMScale:
    """Cross-entropy method for scale of Gaussian distribution
    p(x) ~ N(mu, Sigma)
    mu is known
    Sigma is diagonal and known up to a constant, Sigma = sigma*Lambda

    x is in [0,1]**num_dim
    """

    def __init__(self, mu, cov_aspect, sigma_0, alpha=1.0):
        self.num_dim = len(mu)

        cov_aspect = np.abs(cov_aspect)
        cov_aspect = cov_aspect / cov_aspect.mean()
        det = np.prod(cov_aspect)
        self._unit_cov_diag = cov_aspect / (det ** (1 / self.num_dim))
        self._df = 1
        self._mu = mu
        self._scale2 = sigma_0**2
        self._alpha = alpha

    def sigma(self):
        return np.sqrt(self._scale2)

    def ask(self, num_samples, qmc=False):
        rv = ScaledInverseChi2(self._df, self._scale2)
        scale2 = rv.rvs(size=(1,))
        cov = scale2 * self._unit_cov_diag

        return draw_bounded_normal_samples(self._mu, cov, num_samples, qmc)

    def tell(self, x, pi, p_max):
        x = np.asarray(x)
        pi = np.asarray(pi)

        dx = (x - self._mu) / self._unit_cov_diag
        w = 1 / pi

        score = p_max / pi

        if True:
            i = np.where(score > np.median(score))[0]
            dx = dx[i, :]
            w = w[i]

        d2 = (w[:, None] * dx * dx).sum(axis=0) / w.sum()
        scale2_est = d2.mean()

        if self._alpha is None:
            self._df += len(w)
            self._scale2 += scale2_est
        else:
            self._df += self._alpha * (len(w) - self._df)
            self._scale2 += self._alpha * (scale2_est - self._scale2)
