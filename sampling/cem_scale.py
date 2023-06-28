from dataclasses import dataclass

import numpy as np
from scipy.stats import multivariate_normal

from sampling.scaled_inv_chi2 import ScaledInverseChi2


@dataclass
class _CEMSample:
    prob: np.array
    x: np.array


class CEMScale:
    """Cross-entropy method for scale of Gaussian distribution
    p(x) ~ N(mu, Sigma)
    mu is known
    Sigma is diagonal and known up to a constant, Sigma = sigma*Lambda

    x is in [0,1]**num_dim
    """

    def __init__(self, mu, unit_cov_diag, sigma_0, alpha=0.2):
        self.num_dim = len(mu)
        assert unit_cov_diag.shape == mu.shape, (unit_cov_diag.shape, mu.shape)
        det = np.prod(unit_cov_diag)
        assert (det - 1) < 1e-6, det
        self._df = 1
        self._loc = mu
        self._unit_cov_diag = unit_cov_diag
        self._scale2 = sigma_0**2
        self._alpha = alpha

    def estimate_sigma(self):
        return np.sqrt(self._scale2)

    def ask(self, num_samples):
        samples = []
        while len(samples) < num_samples:
            rv = ScaledInverseChi2(self._df, self._scale2)
            scale2s = rv.rvs(size=(num_samples,))

            for scale2 in scale2s:
                cov = scale2 * self._unit_cov_diag
                rv_norm = multivariate_normal(
                    mean=self._loc,
                    cov=cov,
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

        return samples[:num_samples]

    def tell(self, likelihoods, samples, n_keep=None):
        likelihoods = np.asarray(likelihoods)
        dx = np.stack([s.x for s in samples]) - self._loc
        probs = np.array([s.prob for s in samples])

        probs = probs / probs.sum()

        # importance-weighted log-likelihood: log( p(x|theta) / p(x) )
        # Careful: Adding eps to probs might break this.
        scores = np.log(likelihoods) - np.log(probs)
        i = scores.argsort()
        if n_keep is None:
            n_keep = len(dx) // 2

        i = i[-n_keep:].tolist()
        dx = dx[i, :]
        probs = probs[i]

        d2 = (dx * dx).mean(axis=0)
        scale2_est = d2.mean()

        self._df += self._alpha * (n_keep - self._df)
        self._scale2 += self._alpha * (scale2_est - self._scale2)
