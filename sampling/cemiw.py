import numpy as np
from scipy.stats import invwishart, multivariate_normal

# TODO: num_dim > 1
# TODO: unit test
# TODO: maybe QMCSampler instead of np.random.normal()


class CEMIW:
    """Cross-entropy method for inverse-Wishart distribution
    p(x) ~ N(mu, Sigma)
    with known mu, find Sigma.
    x is a vector, Sigma is the (diagnoal) covariance matrix.
    """

    def __init__(self, mu, scale_0, alpha=0.2):
        self.mu = mu
        self.num_dim = len(mu)
        self._df = 1
        self._scale = scale_0 * np.eye(self.num_dim)
        self._alpha = alpha

    def ask(self, num_samples):
        rv = invwishart(df=self._df, scale=self._scale)
        sigma_squareds = rv.rvs(size=(num_samples,))
        sigmas = np.sqrt(sigma_squareds)

        xs = self.mu + sigmas * np.random.normal(size=(len(sigmas),))
        if len(self.mu) == 1:
            xs = xs[:, None]
        else:
            xs = np.atleast_2d(xs)
        probs = np.array([multivariate_normal.pdf(x, mean=self.mu, cov=sigma**2) for x, sigma in zip(xs, sigmas)])

        return sigmas, probs, xs

    def _log_prob(self, p):
        eps = 0.0  #  Leave it alone, or sigma will explode when the true sigma is small.
        return np.log(eps + (1 - 2 * eps) * p)

    def tell(self, likelihoods, sigmas, probs, xs, n_keep=None):
        x_minus_mu = xs - self.mu
        del xs

        # importance-weighted log-likelihood: log( p(x|theta) / p(x) )
        scores = self._log_prob(likelihoods.flatten()) - self._log_prob(probs)
        x_minus_mu = x_minus_mu[scores.argsort(), :]
        if n_keep is None:
            n_keep = len(x_minus_mu) // 2
        xs_keep = x_minus_mu[-n_keep:, :]

        self._df += self._alpha * (n_keep - self._df)
        self._scale += self._alpha * (np.diag(xs_keep.T @ xs_keep) - self._scale)
