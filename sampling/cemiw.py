from dataclasses import dataclass

import numpy as np
from scipy.stats import invwishart, multivariate_normal


@dataclass
class _CEMSample:
    cov: np.array
    prob: np.array
    x: np.array


class CEMIW:
    """Cross-entropy method for inverse-Wishart distribution
    p(x) ~ N(mu, Sigma)
    with known mu, find Sigma.
    x is a vector, Sigma is the (diagnoal) covariance matrix.
    """

    def __init__(self, mu, scale_0, alpha=0.2):
        self.mu = mu
        self.num_dim = len(mu)
        self._df = self.num_dim
        self._scale = scale_0 * np.eye(self.num_dim)
        self._alpha = alpha

    def ask(self, num_samples):
        rv_iw = invwishart(df=self._df, scale=self._scale)
        covs = rv_iw.rvs(size=(num_samples,))

        samples = []
        for cov in covs:
            if isinstance(cov, np.ndarray):
                cov = np.diag(np.diag(cov))
            rv_norm = multivariate_normal(mean=self.mu, cov=cov)
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
        x_minus_mu = np.stack([s.x for s in samples]) - self.mu
        if self.num_dim == 1:
            x_minus_mu = x_minus_mu[:, None]
        probs = np.array([s.prob for s in samples])

        # importance-weighted log-likelihood: log( p(x|theta) / p(x) )
        scores = self._log_prob(likelihoods) - self._log_prob(probs)
        x_minus_mu = x_minus_mu[scores.argsort(), :]
        if n_keep is None:
            n_keep = len(x_minus_mu) // 2
        xs_keep = x_minus_mu[-n_keep:, :]

        self._df += self._alpha * (n_keep - self._df)
        self._scale += self._alpha * (np.diag(np.diag(xs_keep.T @ xs_keep)) - self._scale)
