import numpy as np

from .util import mk_normal_samples


class PXMax:
    def __init__(self, mu, cov_aspect, sigma_0):
        self._mu = mu
        self._num_dim = len(mu)
        assert len(mu) == len(cov_aspect), (
            "cov_aspect should be a vector, representing a diagonal covariance matrix, and it should have the same length as mu",
            len(mu),
            len(cov_aspect),
        )
        adet = np.abs(np.prod(cov_aspect))
        self._unit_cov_diag = cov_aspect / (adet ** (1 / self._num_dim))
        self._sigma = sigma_0

    def ask(self, num_samples):
        cov = (self._sigma**2) * self._unit_cov_diag
        return mk_normal_samples([(self._mu, cov)], num_samples)

    # TODO
    # - sigma2 = sigma_0*2
    # - START: generate (pi_i, x_i) samples from MVN( mu, (sigma2)*unit_cov_diag)
    # - pi_i is probability from generating distribution; importance weight
    # - sample x_max_i = argmax f({x_i}); redo N times to get N samples from x_max_i
    #    -- keep corresponding pi_i with x_max_i
    # - calculate sigma2 = 1/pi-weighted estimate of avg. variance; see scale2_est in sampling/cem_scale.py
    # - goto START
    # - don't discard any points; keep them around and with their pi_i

    # Mixing importance weights
    #  N p_i
    #  M q_i
    #  sum(p_i) == sum(q_i) == 1
    #
    # p_i <- (N p_i) / (N + M)
    # q_i <- (M q_i) / (N + M)
    #
