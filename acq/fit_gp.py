from typing import Any

import gpytorch
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Warp
from botorch.optim.closures.core import ForwardBackwardClosure
from botorch.optim.utils import get_parameters
from botorch.utils import standardize
from gpytorch.kernels import RFFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood
from gpytorch.priors.torch_priors import LogNormalPrior, NormalPrior
from torch import Tensor
from torch.nn import Module

import common.all_bounds as all_bounds
from acq.sal_transform import SALTransform
from acq.y_transform import YTransform
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


def standardize_torch(Y):
    # TODO: Standardize Yvar, too
    assert len(Y.shape) == 2, Y.shape
    if len(Y) == 0:
        return Y
    if len(Y) == 1:
        return 0 * Y

    # Y = torch.sign(Y) * torch.log(1 + torch.abs(Y))
    return standardize(Y)


# def standardize_np(y: np.ndarray):
#     assert len(y.shape) == 2, y.shape
#     assert y.shape[-1] == 1, y.shape

#     if len(y) == 0:
#         return y
#     if len(y) == 1:
#         return 0 * y

#     norm = y.std(axis=0)
#     norm[norm == 0] = 1
#     y = (y - y.mean(axis=0)) / norm
#     y[norm == 0] = 0
#     return y


def _parse_spec(model_spec):
    model_type = None
    input_warping = None
    output_warping = None
    # VanillaBO lengthscale prior is now the default in BoTorch
    model_types = {"gp", "rff8", "rff128", "rff256", "rff512", "rff1024", "dumbo", "rdumbo"}

    if model_spec is not None:
        for s in model_spec.split("+"):
            if s in model_types:
                assert model_type is None, (model_type, model_spec)
                model_type = s
            elif s == "wi":
                assert input_warping is None, (input_warping, model_spec)
                input_warping = True
            elif s == "wos":
                assert output_warping is None, (output_warping, model_spec)
                output_warping = "sal"
            elif s == "woy":
                assert output_warping is None, (output_warping, model_spec)
                output_warping = "y"
            else:
                assert False, ("Unknown option", s)

    if model_type is None:
        model_type = "gp"
    if input_warping is None:
        input_warping = False
    if output_warping is None:
        output_warping = "none"
    # print(f"MODEL_SPEC: model_type = {model_type} input_warping = {input_warping} output_warping = {output_warping}")
    return model_type, input_warping, output_warping


def get_closure(mll, outcome_warp):
    def closure_warping(**kwargs: Any) -> Tensor:
        model = mll.model
        model_output = model(*model.train_inputs)
        warped_inputs = (model.transform_inputs(X=t_in) for t_in in model.train_inputs)
        warped_targets = outcome_warp(model.train_targets)
        log_likelihood = mll(
            model_output,
            warped_targets,
            *warped_inputs,
            **kwargs,
        )
        return -log_likelihood

    return ForwardBackwardClosure(
        forward=closure_warping,
        backward=Tensor.backward,
        parameters=get_parameters(mll, requires_grad=True),
        reducer=Tensor.sum,
        context_manager=None,
    )


def fit_gp_XY(X, Y, model_spec=None):
    model_type, input_warping, output_warping = _parse_spec(model_spec)
    del model_spec

    X = X.to(dtype=torch.double)
    Y = Y.to(dtype=torch.double)

    if len(X) == 0:
        if model_type == "dumbo":
            gp = DUMBOGP(X, Y, use_rank_distance=False)
        elif model_type == "rdumbo":
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
            concentration1_prior=LogNormalPrior(0.0, 1.0),
            concentration0_prior=LogNormalPrior(0.0, 1.0),
        ).to(X)
    else:
        input_transform = None

    if output_warping == "sal":
        outcome_warp = SALTransform(
            a_prior=NormalPrior(0, 1),
            b_prior=LogNormalPrior(0.0, 1.0),
            c_prior=LogNormalPrior(0.0, 1.0),
            d_prior=NormalPrior(0, 1),
        ).to(X)
    elif output_warping == "y":
        outcome_warp = YTransform(
            a_prior=LogNormalPrior(0.0, 0.1),
            b_prior=NormalPrior(0, 1),
        ).to(X)
    else:
        assert output_warping == "none", output_warping
        outcome_warp = None

    Y = standardize_torch(Y).to(X)

    a = torch.tensor(1.0, dtype=torch.double)
    a.requires_grad = True

    if model_type == "dumbo":
        assert input_transform is None, "Unsupported"
        return DUMBOGP(X, Y, use_rank_distance=False)
    elif model_type == "rdumbo":
        assert input_transform is None, "Unsupported"
        return DUMBOGP(X, Y, use_rank_distance=True)
    elif model_type.startswith("rff"):
        num_samples = int(model_type[3:])
        print(f"Using RFF({num_samples})")
        gp = SingleTaskGP(
            X,
            Y,
            input_transform=input_transform,
            covar_module=ScaleKernel(RFFKernel(ard_num_dims=X.shape[-1], num_samples=num_samples)),
        )
    else:
        gp = SingleTaskGP(X, Y, input_transform=input_transform)

    if outcome_warp:
        gp.outcome_warp = outcome_warp

    gp.to(X)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

    # See TuRBO code

    max_cholesky_size = 2000
    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        m = None
        for i_try in range(1):
            mll.to(X)
            try:
                fit_gpytorch_mll(
                    mll,
                    closure=get_closure(mll, outcome_warp) if outcome_warp else None,
                )
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


def mk_policies(policy_0, X_cand):
    policies = []
    for x in X_cand:
        policy = policy_0.clone()
        x = (x.detach().cpu().numpy().flatten() - all_bounds.bt_low) / all_bounds.bt_width
        p = all_bounds.p_low + all_bounds.p_width * x
        policy.set_params(p)
        policies.append(policy)
    return policies
