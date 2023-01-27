import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from bo.standardizer import Standardizer


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
