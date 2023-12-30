import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood

import common.all_bounds as all_bounds


def fit_gp(data, dtype=torch.float64):
    Y, X = zip(*[mk_yx(d) for d in data])
    Y = torch.tensor(Y)[:, None]
    X = torch.stack(X).type(dtype)
    Y = standardize(Y).type(dtype)
    gp = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    m = None
    for i_try in range(3):
        try:
            fit_gpytorch_mll(mll)
        except (RuntimeError, ModelFittingError) as e:
            m = e
            print(f"Retrying fit i_try = {i_try}")
            print("Trying LeaveOneOutPseudoLikelihood")
            mll = LeaveOneOutPseudoLikelihood(gp.likelihood, gp)
            pass
        else:
            break
    else:
        raise m
    return gp, Y, X


def mk_yx(datum):
    return datum.trajectory.rreturn, mk_x(datum.policy)


def mk_x(policy):
    return all_bounds.bt_low + all_bounds.bt_width * ((torch.as_tensor(policy.get_params()) - all_bounds.p_low) / all_bounds.p_width)


def estimate(gp, X):
    return gp.posterior(X).mean.squeeze(-1)
