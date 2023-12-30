import numpy as np

from bo.fit_gp import fit_gp, mk_x


class ArmBestEst:
    def __init__(self):
        print("Using ArmBestEst")
        self._Y_hat = None

    def _fit(self, data):
        gp, Y, X = fit_gp(data)
        self._X = X.detach().numpy()
        self._Y_hat = gp.posterior(X, observation_noise=False).mean.squeeze(-1).detach().numpy()

    def _get_est(self, datum):
        # slow, silly, but works
        x = mk_x(datum.policy)
        dists = []
        for X in self._X:
            d = ((x - X) ** 2).sum()
            dists.append(float(d))
        dists = np.array(dists)
        i = np.random.choice(np.where(dists == min(dists))[0])
        assert dists[i] < 1e-6, (x, dists[i], dists)
        return self._Y_hat[i]

    def __call__(self, data):
        self._fit(data)

        y_hat = np.array([self._get_est(d) for d in data])
        i = np.random.choice(np.where(y_hat == y_hat.max())[0])

        return data[i].policy, y_hat[i]
