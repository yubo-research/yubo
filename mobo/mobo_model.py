import torch


class MOBOModel:
    def __init__(self, kernel, measurements: torch.Tensor, obs_obs_distances: torch.Tensor):
        ds = obs_obs_distances.shape
        assert len(ds) == 2 and ds[0] == ds[1], ds
        assert ds[0] == len(measurements), (ds, len(measurements))

        measurements = measurements.detach()
        obs_obs_distances = obs_obs_distances.detach()
        self._kernel = kernel
        self._mu_0 = measurements.mean()
        if len(measurements) == 1:
            self._std_0 = 1
        elif len(measurements) == 2:
            self._std_0 = torch.abs(measurements[0] - measurements[1])
        else:
            self._std_0 = measurements.std(unbiased=True)
        self._y = (measurements - self._mu_0) / (1e-9 + self._std_0)
        self._obs_obs_kernel = self._kernel(obs_obs_distances) + 0.01 * torch.eye(len(self._y))
        # TODO: Cholesky
        # self._Koo_chol = torch.linalg.cholesky(self._obs_obs_kernel)
        self._Koo_inv = torch.linalg.pinv(self._obs_obs_kernel)
        self._Koo_inv_y = torch.linalg.lstsq(self._obs_obs_kernel, self._y)[0]
        # self._Koo_inv_y = torch.cholesky_solve(torch.atleast_2d(self._y), self._Koo_chol)[0]

    def predict(self, obs_pred_distances: torch.Tensor, pred_pred_distances: torch.Tensor):
        y = self._y
        Kop = self._kernel(obs_pred_distances)
        Kpp = self._kernel(pred_pred_distances)

        if True:
            Kop_T_Koo_inv = Kop.T @ self._Koo_inv
            mu = Kop_T_Koo_inv @ y
            cov = Kpp - Kop_T_Koo_inv @ Kop
        else:
            mu = Kop.T @ self._Koo_inv_y
            # cov = Kpp - Kop.T @ torch.linalg.lstsq(self._obs_obs_kernel, Kop)[0]
            cov = Kpp - Kop.T @ torch.cholesky_solve(Kop, self._Koo_chol)[0]

        # TODO: figure out why cov goes negative, and fix it
        # min_cov = cov.min().item()
        # assert min_cov >= -.1, min_cov
        cov = torch.max(torch.tensor(0.0), cov)

        mu = self._std_0 * mu + self._mu_0
        cov = self._std_0**2 * cov
        return mu, cov
