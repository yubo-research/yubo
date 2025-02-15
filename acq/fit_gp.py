import numpy as np
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Warp
from botorch.utils import standardize

# from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel  # , ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior
from torch.nn import Module

import common.all_bounds as all_bounds
from model.dumbo import DUMBOGP


class _EmptyTransform(Module):
    def __init__(self):
        super().__init__()

    def forward(self, Y, Yvar=None):
        return Y, Yvar

    def untransform(self, Y, Yvar=None):
        return Y, Yvar

    def untransform_posterior(self, posterior):
        return posterior


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


def _parse_spec(model_spec):
    model_type = None
    input_warping = None
    model_types = {"gp", "dumbo", "rdumbo", "vanilla"}

    if model_spec is not None:
        for s in model_spec.split("+"):
            if s in model_types:
                assert model_type is None, (model_type, s)
                model_type = s
            elif s == "wi":
                assert input_warping is None, (input_warping, s)
                input_warping = True

    if model_type is None:
        model_type = "gp"
    if input_warping is None:
        input_warping = False
    return model_type, input_warping


def fit_gp_XY(X, Y, model_spec=None):
    model_type, input_warping = _parse_spec(model_spec)

    if len(X) == 0:
        if model_spec == "dumbo":
            gp = DUMBOGP(X, Y, use_rank_distance=False)
        elif model_spec == "rdumbo":
            gp = DUMBOGP(X, Y, use_rank_distance=True)
        else:
            gp = SingleTaskGP(X, Y, outcome_transform=_EmptyTransform())
        gp.to(X)
        gp.eval()
        return gp

    if input_warping:
        # See http://proceedings.mlr.press/v32/snoek14.pdf
        # See https://botorch.org/docs/tutorials/bo_with_warped_gp/
        input_transform = Warp(
            indices=list(range(X.shape[-1])),
            # use a prior with median at 1.
            # when a=1 and b=1, the Kumaraswamy CDF is the identity function
            concentration1_prior=LogNormalPrior(0.0, 0.75**0.5),
            concentration0_prior=LogNormalPrior(0.0, 0.75**0.5),
        )
    else:
        input_transform = None

    Y = standardize(Y).to(X)

    if model_spec == "vanilla":
        num_dims = X.shape[-1]
        _gp = SingleTaskGP(X, Y, input_transform=input_transform)
        gp = SingleTaskGP(X, Y, covar_module=get_vanilla_kernel(num_dims, _gp._aug_batch_shape), input_transform=input_transform)
    elif model_spec == "dumbo":
        assert input_transform is None, "Unsupported"
        return DUMBOGP(X, Y, use_rank_distance=False)
    elif model_spec == "rdumbo":
        assert input_transform is None, "Unsupported"
        return DUMBOGP(X, Y, use_rank_distance=True)
    else:
        gp = SingleTaskGP(X, Y, input_transform=input_transform)
    gp.to(X)

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

    m = None
    for i_try in range(3):
        mll.to(X)
        try:
            fit_gpytorch_mll(mll)
            # fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 10}})
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
