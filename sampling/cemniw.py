from dataclasses import dataclass

import numpy as np
from scipy.stats import invwishart, multivariate_normal


@dataclass
class _CEMSample:
    cov: np.array
    prob: np.array
    x: np.array


class CEMNIW:
    """Cross-entropy method for inverse-Wishart distribution
    p(x) ~ N(mu, Sigma)
    with known mu, find Sigma.
    x is a vector, Sigma is the (diagnoal) covariance matrix.
    """

    def __init__(self, mu_0, scale_0, alpha=0.2, known_mu=False):
        self.num_dim = len(mu_0)
        self._df = self.num_dim
        self._loc = mu_0
        self._scale = scale_0 * np.eye(self.num_dim)
        self._alpha = alpha
        self._known_mu = known_mu

    def estimate_mu_cov(self):
        return self._loc, np.diag(self._scale/self._df)
        
    def ask(self, num_samples):
        rv_iw = invwishart(df=self._df, scale=self._scale)
        covs = rv_iw.rvs(size=(num_samples,))

        samples = []
        for cov in covs:
            if isinstance(cov, np.ndarray):
                cov = np.diag(np.diag(cov))
            rv_norm = multivariate_normal(mean=self._loc, cov=cov)
            x = rv_norm.rvs(size=(1,))
            samples.append(
                _CEMSample(
                    cov=cov,
                    prob=rv_norm.pdf(x),
                    x=x,
                )
            )

        return samples

    def _log_prob(self, p):
        # TODO: remove
        eps = 0.0  #  Leave it alone, or sigma will explode when the true sigma is small.
        return np.log(eps + (1 - 2 * eps) * p)

    def tell(self, likelihoods, samples, n_keep=None):
        likelihoods = np.asarray(likelihoods)
        dx = np.stack([s.x for s in samples]) - self._loc
        if self.num_dim == 1:
            dx = dx[:, None]
        probs = np.array([s.prob for s in samples])

        # importance-weighted log-likelihood: log( p(x|theta) / p(x) )
        scores = self._log_prob(likelihoods) - self._log_prob(probs)
        dx = dx[scores.argsort(), :]
        if n_keep is None:
            n_keep = len(dx) // 2
        dx_keep = dx[-n_keep:, :]

        self._df += self._alpha * (n_keep - self._df)
        if not self._known_mu:
            self._loc += self._alpha * dx_keep.mean(axis=0)
        self._scale += self._alpha * (np.diag(np.diag(dx_keep.T @ dx_keep)) - self._scale)
