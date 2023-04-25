from dataclasses import dataclass

import numpy as np
from scipy.stats import invwishart, multivariate_normal, qmc


@dataclass
class _CEMSample:
    prob: np.array
    x: np.array


class CEMNIW:
    """Cross-entropy method for inverse-Wishart distribution
    p(x) ~ N(mu, Sigma)
    with known mu, find Sigma.
    x is a vector, Sigma is the (diagnoal) covariance matrix.

    x is in [0,1]**num_dim
    """

    def __init__(self, mu_0, scale_0, alpha=0.2, known_mu=False, sobol_seed=None):
        self.num_dim = len(mu_0)
        self._df = self.num_dim
        self._loc = mu_0
        self._scale = scale_0 * np.eye(self.num_dim)
        self._alpha = alpha
        self._known_mu = known_mu
        self._init = True
        self._sobol_seed = sobol_seed

    def estimate_mu_cov(self):
        return self._loc, np.diag(self._scale / self._df)

    def _init_sample(self, num_samples):
        samples = []
        prob = 1.0 / num_samples
        for x in qmc.Sobol(self.num_dim, seed=self._sobol_seed).random_base2(int(np.log2(num_samples)) + 1)[:num_samples]:
            samples.append(_CEMSample(prob=prob, x=x))
        self._init = False
        return samples

    def ask(self, num_samples):
        if self._init:
            return self._init_sample(num_samples)

        samples = []
        b_done = False
        while not b_done:
            rv_iw = invwishart(df=self._df, scale=self._scale)
            covs = rv_iw.rvs(size=(num_samples,))

            for cov in covs:
                if isinstance(cov, np.ndarray):
                    cov = np.diag(np.diag(cov))
                rv_norm = multivariate_normal(
                    mean=self._loc,
                    cov=1e-9 * np.eye(cov.shape[0]) + cov,
                )
                x = rv_norm.rvs(size=(1,))
                if self.num_dim == 1:
                    x = np.array([x])
                if x.min() < 0 or x.max() > 1:
                    continue
                samples.append(
                    _CEMSample(
                        prob=rv_norm.pdf(x),
                        x=x,
                    )
                )
                if len(samples) == num_samples:
                    b_done = True
                    break

        return samples

    def tell(self, likelihoods, samples, n_keep=None):
        likelihoods = np.asarray(likelihoods)
        dx = np.stack([s.x for s in samples]) - self._loc
        probs = np.array([s.prob for s in samples])

        # importance-weighted log-likelihood: log( p(x|theta) / p(x) )
        # Careful. Adding eps's might break it.
        scores = np.log(likelihoods) - np.log(probs)
        dx = dx[scores.argsort(), :]
        if n_keep is None:
            n_keep = len(dx) // 2
        dx_keep = dx[-n_keep:, :]

        self._df += self._alpha * (n_keep - self._df)
        if not self._known_mu:
            self._loc += self._alpha * dx_keep.mean(axis=0)
        self._scale += self._alpha * (np.diag(np.diag(dx_keep.T @ dx_keep)) - self._scale)
