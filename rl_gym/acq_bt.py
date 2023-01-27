import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood


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


class AcqBT:
    def __init__(self, acq_factory, data):
        Y, X = zip(*[self._mk_yx(d) for d in data])
        Y = torch.tensor(Y)[:, None]

        X = torch.stack(X).type(torch.float64)
        Y = Standardizer(Y)(Y).type(torch.float64)

        gp = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        num_dim = X[0].shape[-1]
        self.acq_function = acq_factory(gp)
        self.bounds = torch.tensor([[0.0] * num_dim, [1.0] * num_dim], device=X.device, dtype=X.dtype)

    def _mk_yx(self, datum):
        return datum.trajectory.rreturn, self._mk_x(datum.policy)

    def _mk_x(self, policy):
        return torch.as_tensor(policy.get_params())

    def __call__(self, policy):
        X = torch.atleast_2d(self._mk_x(policy))
        X = X.unsqueeze(0)
        return self.acq_function(X).squeeze().item()
