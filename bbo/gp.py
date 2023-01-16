import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# TODO
# After the full optimizer works, replace this with GPyTorch (or BoTorch).
# Then you can switch to a Matern kernel and qNEI acquisition function.
# DON'T develop this any further.


# from https://github.com/swyoon/pytorch-minimal-gaussian-process
class GP(nn.Module):
    def __init__(self, length_scale=1.0, noise_scale=1.0, amplitude_scale=1.0):
        super().__init__()
        self._length_scale = nn.Parameter(torch.tensor(np.log(length_scale)))
        self._noise_scale = nn.Parameter(torch.tensor(np.log(noise_scale)))
        self._amplitude_scale = nn.Parameter(torch.tensor(np.log(amplitude_scale)))
        self._min_noise = 1e-6

    def _bound(self, scale):
        return (1 + torch.tanh(scale)) / 2

    @property
    def length_scale(self):
        return self._bound(self._length_scale)

    @property
    def noise_scale(self):
        return self._min_noise + self._bound(self._noise_scale)

    @property
    def amplitude_scale(self):
        return self._bound(self._amplitude_scale)

    def forward(self, dist_obs_pred, dist_pred_pred=None):
        """compute prediction. fit() must have been called."""
        # from IPython.core.debugger import set_trace
        dist_obs_pred = torch.as_tensor(dist_obs_pred)
        if dist_pred_pred is None:
            simple = True
            dist_pred_pred = torch.tensor([0.0])
        else:
            simple = False
            dist_pred_pred = torch.as_tensor(dist_pred_pred)

        k_obs_pred = self._kernel(dist_obs_pred)
        k_pred_pred = self._kernel(dist_pred_pred)
        v = torch.linalg.solve(self._L, k_obs_pred)
        mu = self._y_mu + self._y_std * k_obs_pred.T.mm(self._alpha)
        # var = self._y_std**2 * (self.amplitude_scale + self.noise_scale - torch.diag(v.T.mm(v)))
        cov = self._y_std**2 * (k_pred_pred - v.T.mm(v))
        if simple:
            mu = mu.item()
            cov = cov.item()
        return mu, cov

    def fit(self, dist_obs_obs, y):
        """should be called before forward() call."""
        y_mu = y.mean()
        if len(y) > 1:
            y_std = 1e-9 + y.std()
        else:
            y_std = 1
        y_use = (y - y_mu) / y_std

        K = self._kernel(dist_obs_obs) + self.noise_scale * torch.eye(dist_obs_obs.shape[0])
        try:
            L = torch.linalg.cholesky(K)
        except torch._C._LinAlgError:
            return None
        alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, y_use))
        marginal_likelihood = -0.5 * y_use.T.mm(alpha) - torch.log(torch.diag(L)).sum()  # just a constant - D * 0.5 * np.log(2 * np.pi)
        self._y_mu = y_mu
        self._y_std = y_std
        self._L = L
        self._alpha = alpha
        return marginal_likelihood

    def _kernel(self, dist):
        return self.amplitude_scale * torch.exp(-0.5 * (dist**2) / self.length_scale)

    def train(self, dist_obs_obs, y):
        dist_obs_obs = torch.as_tensor(dist_obs_obs)
        y = torch.as_tensor(y)
        opt = optim.Adam(self.parameters(), lr=0.1)
        for _ in range(1000):
            self._train_step(dist_obs_obs, y, opt)

    def _train_step(self, dist_obs_obs, y, opt):
        """gradient-based optimization of hyperparameters
        opt: torch.optim.Optimizer object."""
        opt.zero_grad()
        fit = None
        for _ in range(10):
            fit = self.fit(dist_obs_obs, y)
            if fit is not None:
                break
            self._min_noise *= 10
        if fit is None:
            raise Exception("FAILED TO FIT")
        nll = -fit.sum()
        nll.backward()
        opt.step()

        # return {
        #    "loss": nll.item(),
        #    "length": self.length_scale.detach().cpu(),
        #    "noise": self.noise_scale.detach().cpu(),
        #    "amplitude": self.amplitude_scale.detach().cpu(),
        # }
