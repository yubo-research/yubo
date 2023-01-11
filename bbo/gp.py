import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def distance_matrix(X, Z):
    Xsq = (X**2).sum(dim=1, keepdim=True)
    Zsq = (Z**2).sum(dim=1, keepdim=True)
    return Xsq + Zsq.T - 2 * X.mm(Z.T)


# from https://github.com/swyoon/pytorch-minimal-gaussian-process
class GP(nn.Module):
    def __init__(self, distance_matrix_fn, length_scale=1.0, noise_scale=1.0, amplitude_scale=1.0):
        super().__init__()
        self._distance_matrix_fn = distance_matrix_fn
        self._length_scale = nn.Parameter(torch.tensor(np.log(length_scale)))
        self._noise_scale = nn.Parameter(torch.tensor(np.log(noise_scale)))
        self._amplitude_scale = nn.Parameter(torch.tensor(np.log(amplitude_scale)))

    def _bound(self, scale):
        return (1 + torch.tanh(scale)) / 2

    @property
    def length_scale(self):
        return self._bound(self._length_scale)

    @property
    def noise_scale(self):
        return 1e-6 + self._bound(self._noise_scale)

    @property
    def amplitude_scale(self):
        return self._bound(self._amplitude_scale)

    def forward(self, x):
        """compute prediction. fit() must have been called."""
        L = self.L
        alpha = self.alpha
        k = self.kernel_mat(self.X, x)
        v = torch.linalg.solve(L, k)
        mu = self._y_mu + self._y_std * k.T.mm(alpha)
        var = self._y_std**2 * (self.amplitude_scale + self.noise_scale - torch.diag(v.T.mm(v)))
        return mu, var

    def fit(self, X, y):
        """should be called before forward() call."""
        y_mu = y.mean()
        if len(y) > 1:
            y_std = 1e-9 + y.std()
        else:
            y_std = 1
        y_use = (y - y_mu) / y_std

        K = self.kernel_mat(X, X) + self.noise_scale * torch.eye(len(X))
        L = torch.linalg.cholesky(K)
        alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, y_use))
        marginal_likelihood = -0.5 * y_use.T.mm(alpha) - torch.log(torch.diag(L)).sum()  # just a constant - D * 0.5 * np.log(2 * np.pi)
        self.X = X
        self._y_mu = y_mu
        self._y_std = y_std
        self.L = L
        self.alpha = alpha
        self.K = K
        return marginal_likelihood

    def kernel_mat(self, X, Z):
        sqdist = self._distance_matrix_fn(X, Z)
        return self.amplitude_scale * torch.exp(-0.5 * sqdist / self.length_scale)

    def train(self, X, y):
        opt = optim.Adam(self.parameters(), lr=0.1)
        for _ in range(1000):
            self.train_step(X, y, opt)

    def train_step(self, X, y, opt):
        """gradient-based optimization of hyperparameters
        opt: torch.optim.Optimizer object."""
        opt.zero_grad()
        nll = -self.fit(X, y).sum()
        nll.backward()
        opt.step()
        return {
            "loss": nll.item(),
            "length": self.length_scale.detach().cpu(),
            "noise": self.noise_scale.detach().cpu(),
            "amplitude": self.amplitude_scale.detach().cpu(),
        }
