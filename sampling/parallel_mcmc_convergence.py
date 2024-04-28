import numpy as np


class ParallelMCMCConvergence:
    """
    Gelman-Rubin is popular, but doesn't work here b/c it
     requires the initial points to be cover the target distribution. Our initial
     points are all the same.
     https://arxiv.org/pdf/1909.11827  Section 2.4
    """

    def __init__(self, rho_c=0.5, m_stat_c=2, v_stat_c=1):
        self._X_prev = None
        self._rho_c = rho_c
        self._m_stat_c = m_stat_c
        self._v_stat_c = v_stat_c

    def converged(self, X):
        if self._X_prev is None:
            self._X_prev = X.copy()
            return False

        if self._independent(X):
            if self._statistically_indistinguishsble(X):
                return True
            self._X_prev = X.copy()

        return False

    def _independent(self, X):
        # Have we run long enough for the autocorrelation to decay?
        # With parallel chains, you can measure the autocorrelation
        #  directly, unlike with a run of a single chain.
        rho = np.corrcoef(
            X.flatten(),
            self._X_prev.flatten(),
        )[0, 1]
        # print("RHO:", rho)
        return rho < self._rho_c

    def _statistically_indistinguishsble(self, X):
        # Has the distribution of the points in the chains stopped changing?
        # If so, we presume that we are at equilibrium and further sampling
        #  will not change the distribution.
        mu_prev = self._X_prev.mean(axis=0)
        mu = X.mean(axis=0)
        d_mu = np.abs(mu - mu_prev)

        sd_prev = self._X_prev.std(axis=0)
        sd = X.std(axis=0)
        se_prev = sd_prev / np.sqrt(len(self._X_prev))
        se = sd / np.sqrt(len(self._X_prev))

        t = d_mu / np.sqrt(se_prev**2 + se**2)
        F = sd**2 / sd_prev**2
        m_stat = t.max()
        v_stat = np.abs(F - 1).max()

        print(f"CONV: m_stat = {m_stat:.4f} v_stat = {v_stat:.4f}")
        if m_stat < self._m_stat_c and v_stat < self._v_stat_c:
            return True
        return False
