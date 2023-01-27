import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from rl_gym.iopt import qIOPT


class Standardizer:
    def __init__(self, Y_orig):
        stddim = -1 if Y_orig.dim() < 2 else -2
        Y_std = Y_orig.std(dim=stddim, keepdim=True)
        self.Y_std = Y_std.where(Y_std >= 1e-9, torch.full_like(Y_std, 1.0))
        self.Y_mu = Y_orig.mean(dim=stddim, keepdim=True)

    def __call__(self, Y_orig):
        return (Y_orig - self.Y_mu) / self.Y_std

    def undo(self, Y):
        return self.Y_mu + self.Y_std * Y


class ACQIOpt:
    def __init__(self, data, num_samples=512):
        Y, X = zip(*[self._mk_yx(d) for d in data])
        Y = torch.tensor(Y)[:, None]

        X = torch.stack(X).type(torch.float64)
        Y = Standardizer(Y)(Y).type(torch.float64)

        print("SHAPES:", X.shape, Y.shape)
        gp = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        self._qiopt = qIOPT(gp, q=1, num_samples=num_samples)

    def _mk_yx(self, datum):
        return datum.trajectory.rreturn, self._mk_x(datum.policy)

    def _mk_x(self, policy):
        return torch.as_tensor(policy.get_params())

    def __call__(self, policy):
        X = torch.atleast_2d(self._mk_x(policy))
        X = X.unsqueeze(0)
        return self._qiopt(X).squeeze().item()
