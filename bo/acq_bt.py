import torch
from botorch.acquisition import PosteriorMean
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

import common.all_bounds as all_bounds


class AcqBT:
    def __init__(self, acq_factory, data, acq_kwargs=None):
        Y, X = zip(*[self._mk_yx(d) for d in data])
        Y = torch.tensor(Y)[:, None]

        X = torch.stack(X).type(torch.float64)
        Y = standardize(Y).type(torch.float64)

        gp = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        num_dim = X[0].shape[-1]
        # All BoTorch stuff is coded to bounds of [0,1]!
        self.bounds = torch.tensor([[0.0] * num_dim, [1.0] * num_dim], device=X.device, dtype=X.dtype)

        if not acq_kwargs:
            kwargs = {}
        else:
            kwargs = dict(acq_kwargs)
        if "X_max" in kwargs:
            kwargs["X_max"] = self._find_max(gp, self.bounds)
        if "best_f" in kwargs:
            kwargs["best_f"] = gp(self._find_max(gp, self.bounds)).mean
        if "X_baseline" in kwargs:
            kwargs["X_baseline"] = X
        if "Y_max" in kwargs:
            kwargs["Y_max"] = gp(self._find_max(gp, self.bounds)).mean
        if "bounds" in kwargs:
            kwargs["bounds"] = self.bounds

        self.acq_function = acq_factory(gp, **kwargs)

    def _find_max(self, gp, bounds):
        x_cand, _ = optimize_acqf(
            acq_function=PosteriorMean(model=gp),
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 10, "maxiter": 200},
        )
        return x_cand

    def _mk_yx(self, datum):
        return datum.trajectory.rreturn, self._mk_x(datum.policy)

    def _mk_x(self, policy):
        return all_bounds.bt_low + all_bounds.bt_width*( (torch.as_tensor(policy.get_params()) - all_bounds.p_low) / all_bounds.p_width )

    def __call__(self, policy):
        X = torch.atleast_2d(self._mk_x(policy)).unsqueeze(0)
        return self.acq_function(X).squeeze().item()
