from typing import Any, NamedTuple

import gpytorch
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.approximate_gp import SingleTaskVariationalGP
from botorch.models.transforms.input import Warp
from botorch.optim.closures.core import ForwardBackwardClosure
from botorch.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
from botorch.optim.utils import get_parameters
from botorch.utils import standardize
from gpytorch.kernels import RFFKernel, ScaleKernel
from gpytorch.mlls import (
    ExactMarginalLogLikelihood,
    LeaveOneOutPseudoLikelihood,
    VariationalELBO,
)
from gpytorch.priors.torch_priors import LogNormalPrior, NormalPrior
from torch import Tensor
from torch.nn import Module

import common.all_bounds as all_bounds
from acq.sal_transform import SALTransform
from acq.y_transform import YTransform


class _ParsedSpec(NamedTuple):
    model_type: str
    input_warping: bool
    output_warping: str


class _FitGpResult(NamedTuple):
    gp: object
    Y: Tensor
    X: Tensor


class _EmptyTransform(Module):
    def __init__(self):
        super().__init__()

    def forward(self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        return Y, Yvar

    def untransform(self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        return Y, Yvar

    def untransform_posterior(self, posterior, X: Tensor | None = None):
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


def _parse_spec(model_spec, num_obs):
    _SPARSE_MIN_OBS = 10
    model_type = None
    input_warping = None
    output_warping = None
    # VanillaBO lengthscale prior is now the default in BoTorch
    model_types = {"gp", "rff8", "rff128", "rff256", "rff512", "rff1024", "sparse"}

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
    if model_type == "sparse" and num_obs < _SPARSE_MIN_OBS:
        model_type = "gp"
    if input_warping is None:
        input_warping = False
    if output_warping is None:
        output_warping = "none"
    # print(f"MODEL_SPEC: model_type = {model_type} input_warping = {input_warping} output_warping = {output_warping}")
    return _ParsedSpec(model_type=model_type, input_warping=bool(input_warping), output_warping=str(output_warping))


def get_closure(mll, outcome_warp):
    def closure_warping(**kwargs: Any) -> Tensor:
        model = mll.model
        model_output = model(*model.train_inputs)
        warped_inputs = tuple(model.transform_inputs(X=t_in) for t_in in model.train_inputs)
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
        parameters=get_parameters(mll, requires_grad=True),
    )


def _create_empty_gp(X, Y, model_type):
    if model_type == "sparse":
        inducing_points = torch.empty((0, X.shape[-1]), dtype=X.dtype, device=X.device)
        gp = SingleTaskVariationalGP(X, inducing_points=inducing_points, outcome_transform=_EmptyTransform())
    else:
        gp = SingleTaskGP(X, Y, outcome_transform=_EmptyTransform())
    gp.to(X)
    gp.eval()
    return gp


def _create_outcome_warp(output_warping, X):
    warp_factories = {
        "sal": lambda: SALTransform(
            a_prior=NormalPrior(0, 1),
            b_prior=LogNormalPrior(0.0, 1.0),
            c_prior=LogNormalPrior(0.0, 1.0),
            d_prior=NormalPrior(0, 1),
        ).to(X),
        "y": lambda: YTransform(a_prior=LogNormalPrior(0.0, 0.1), b_prior=NormalPrior(0, 1)).to(X),
        "none": lambda: None,
    }
    return warp_factories[output_warping]()


def _create_gp_model(X, Y, model_type, input_transform):
    if model_type == "sparse":
        return SingleTaskVariationalGP(X, Y, input_transform=input_transform)
    if model_type.startswith("rff"):
        num_samples = int(model_type[3:])
        return SingleTaskGP(
            X,
            Y,
            input_transform=input_transform,
            covar_module=ScaleKernel(RFFKernel(ard_num_dims=X.shape[-1], num_samples=num_samples)),
        )
    return SingleTaskGP(X, Y, input_transform=input_transform)


def _fit_gp_model(gp, mll, model_type, outcome_warp, X):
    max_cholesky_size = 2000
    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        m = None
        num_tries = 2 if model_type == "sparse" else 1
        for i_try in range(num_tries):
            mll.to(X)
            kwargs = {"closure": get_closure(mll, outcome_warp)} if outcome_warp else {}
            if model_type == "sparse":
                opt, opt_kw = (
                    (
                        fit_gpytorch_mll_scipy,
                        {"options": {"maxiter": 4000, "maxfun": 4000}},
                    )
                    if i_try == 0
                    else (fit_gpytorch_mll_torch, {"step_limit": 4000})
                )
                kwargs = dict(kwargs)
                kwargs.update({"optimizer": opt, "optimizer_kwargs": opt_kw})
            try:
                fit_gpytorch_mll(mll, **kwargs)
            except (RuntimeError, ModelFittingError) as e:
                m = e
                print(f"Retrying fit i_try = {i_try}")
                print("Trying LeaveOneOutPseudoLikelihood")
                if model_type != "sparse":
                    mll = LeaveOneOutPseudoLikelihood(gp.likelihood, gp)
            else:
                break
        else:
            raise m


def fit_gp_XY(X, Y, model_spec=None):
    model_type, input_warping, output_warping = _parse_spec(model_spec, num_obs=len(Y))

    if len(X) == 0:
        return _create_empty_gp(X, Y, model_type)

    input_transform = (
        Warp(
            indices=list(range(X.shape[-1])),
            concentration1_prior=LogNormalPrior(0.0, 1.0),
            concentration0_prior=LogNormalPrior(0.0, 1.0),
        ).to(X)
        if input_warping
        else None
    )

    outcome_warp = _create_outcome_warp(output_warping, X)
    Y = standardize_torch(Y).to(X)
    gp = _create_gp_model(X, Y, model_type, input_transform)

    if outcome_warp:
        gp.outcome_warp = outcome_warp
    gp.to(X)

    mll = VariationalELBO(gp.likelihood, gp.model, num_data=X.shape[-2]) if model_type == "sparse" else ExactMarginalLogLikelihood(gp.likelihood, gp)
    _fit_gp_model(gp, mll, model_type, outcome_warp, X)
    return gp


def extract_X_Y(data, dtype, device):
    Y, X = zip(*[mk_yx(d) for d in data])
    Y = torch.tensor(Y)[:, None].type(dtype).to(device)
    X = torch.stack(X).type(dtype).to(device)
    return Y, X


def fit_gp(data, dtype=torch.float64, device="cpu"):
    Y, X = extract_X_Y(data, dtype, device)
    gp = fit_gp_XY(X, Y)
    return _FitGpResult(gp=gp, Y=Y, X=X)


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
