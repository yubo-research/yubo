import numpy as np

from acq.fit_gp import fit_gp


class ArmBestEst:
    def __init__(self):
        print("Using ArmBestEst")

    def _calc_y_hat(self, data):
        gp, Y, X = fit_gp(data)
        self._X = X.detach().numpy()
        return Y.mean() + Y.std() * gp.posterior(X, observation_noise=False).mean.squeeze(-1).detach().numpy()

    def __call__(self, data):
        Y_hat = self._calc_y_hat(data)

        i = np.random.choice(np.where(Y_hat == Y_hat.max())[0])

        return data[i].policy, Y_hat[i]
