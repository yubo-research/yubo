import numpy as np
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.utils import standardize

# from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel  # , ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood
from gpytorch.priors.torch_priors import GammaPrior

import common.all_bounds as all_bounds


def get_vanilla_kernel(num_dim, batch_shape):
    # See section 5.1 of
    #  Hvarfner, C., Hellsten, E.O., & Nardi, L. (2024). Vanilla Bayesian Optimization Performs Great in High Dimensions. ArXiv, abs/2402.02229.
    #
    length_scale_0 = np.sqrt(num_dim)
    return MaternKernel(
        nu=2.5,
        ard_num_dims=num_dim,
        batch_shape=batch_shape,
        lengthscale_prior=GammaPrior(3.0, 6.0 / length_scale_0),
    )


def fit_gp_XY(X, Y, use_vanilla=False):
    Y = standardize(Y).to(X)
    _gp = SingleTaskGP(X, Y)
    if use_vanilla:
        num_dims = X.shape[-1]
        gp = SingleTaskGP(X, Y, covar_module=get_vanilla_kernel(num_dims, _gp._aug_batch_shape))
    else:
        gp = _gp
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    gp.to(X)
    m = None
    for i_try in range(3):
        mll.to(X)
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

    return gp


def extract_X_Y(data, dtype, device):
    Y, X = zip(*[mk_yx(d) for d in data])
    Y = torch.tensor(Y)[:, None].type(dtype).to(device)
    X = torch.stack(X).type(dtype).to(device)
    return Y, X


def fit_gp(data, dtype=torch.float64, device="cpu"):
    Y, X = extract_X_Y(data, dtype, device)
    gp = fit_gp_XY(X, Y)
    return gp, Y, X


def mk_yx(datum):
    return datum.trajectory.rreturn, mk_x(datum.policy)


def mk_x(policy):
    return all_bounds.bt_low + all_bounds.bt_width * ((torch.as_tensor(policy.get_params()) - all_bounds.p_low) / all_bounds.p_width)


def estimate(gp, X):
    return gp.posterior(X).mean.squeeze(-1)
