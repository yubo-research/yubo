import gpytorch
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood

import acq.fit_gp as fit_gp
import common.all_bounds as all_bounds
from acq.mcmc_bo import TurboState, generate_batch_multiple_tr
from optimizer.designer_asserts import assert_scalar_rreturn
from optimizer.sobol_designer import SobolDesigner


class MCMCBODesigner:
    def __init__(
        self,
        policy,
        num_trust_regions=1,
        num_init=None,
        mcmc_type=None,
        dtype=torch.double,
        device=None,
    ):
        assert num_trust_regions == 1, (
            "NYI: Multiple trust regions",
            num_trust_regions,
        )
        self._turbo_state = None
        self._policy = policy
        self._num_init = num_init
        self._mcmc_type = mcmc_type
        self._dtype = dtype
        self._device = device

    def _sobol(self, num_arms):
        return SobolDesigner(self._policy.clone(), max_points=num_arms)(None, num_arms)

    def __call__(self, data, num_arms, *, telemetry=None):
        num_dim = self._policy.num_params()

        if len(data) < self._num_init:
            return self._sobol(num_arms)

        assert_scalar_rreturn(data)

        if self._turbo_state is None or self._turbo_state.restart_triggered:
            y_max = max(d.trajectory.rreturn for d in data)
            self._turbo_state = TurboState(
                dim=num_dim,
                batch_size=num_arms,
                best_value=y_max,
            )
            self._effective_num_init = len(data) + self._num_init

        if len(data) < self._effective_num_init:
            return self._sobol(num_arms)

        num_candidates = min(5000, max(2000, 200 * num_dim))
        Y, X = fit_gp.extract_X_Y(data, self._dtype, self._device)
        self._turbo_state.update_state(Y)
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = (
            ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=num_dim,
                    lengthscale_constraint=Interval(0.005, 4.0),
                )
            )
        )

        model = SingleTaskGP(
            X,
            Y,
            covar_module=covar_module,
            likelihood=likelihood,
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with gpytorch.settings.max_cholesky_size(float("inf")):
            try:
                fit_gpytorch_mll(mll)
            except (RuntimeError, ModelFittingError):
                print("Trying LeaveOneOutPseudoLikelihood")
                mll = LeaveOneOutPseudoLikelihood(model.likelihood, model)

        X_cand, _ = generate_batch_multiple_tr(
            state=[self._turbo_state],
            model=[model],
            X=X.unsqueeze(0),
            Y=Y.unsqueeze(0),  # add Y_turbo to recover normalized GP sampling
            batch_size=num_arms,
            num_candidates=num_candidates,
            acqf="ts",
            mcmc_type=self._mcmc_type,
            dtype=self._dtype,
            device=self._device,
        )

        policies = []
        for x in X_cand:
            policy = self._policy.clone()
            x = (
                x.detach().cpu().numpy().flatten() - all_bounds.bt_low
            ) / all_bounds.bt_width
            p = all_bounds.p_low + all_bounds.p_width * x
            policy.set_params(p)
            policies.append(policy)
        return policies
