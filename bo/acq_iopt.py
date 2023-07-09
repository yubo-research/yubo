import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
)
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform

# from IPython.core.debugger import set_trace
from torch import Tensor
from torch.quasirandom import SobolEngine


class AcqIOpt(MCAcquisitionFunction):
    def __init__(self, model: Model, acqf: AcquisitionFunction = None, num_X_samples: int = 256, **kwargs) -> None:
        super().__init__(model=model, **kwargs)

        X_0 = self.model.train_inputs[0]
        num_dim = X_0.shape[-1]
        dtype = X_0.dtype

        sobol_engine = SobolEngine(num_dim, scramble=True)
        X_samples = sobol_engine.draw(num_X_samples, dtype=dtype)
        self.register_buffer("X_samples", X_samples)

        if len(X_0) == 0:
            p_iopt = 1.0
        else:
            # The probability that the optimum will be found by iopt
            #  (global search) is proportional to the mean variance.
            # The probability that the optimum will be found by simple
            #  maximization (local search) is 1 - i_opt.
            p_iopt = self._mean_variance(model, num_dim, num_X_samples)

        # TODO: ensemble of arms
        self.acqf = None if np.random.uniform() < p_iopt else acqf

    def _mean_variance(self, model, num_dim, num_X_samples):
        X = torch.rand(size=(num_X_samples, num_dim))
        Y = model.posterior(X)
        return Y.variance.mean().item()

    def model_t(self, X):
        Y = self.model.posterior(X).mean  # b x q x 1
        return self.model.condition_on_observations(X=X, Y=Y)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        if self.acqf is not None:
            return self.acqf(X)

        model_t = self.model_t(X)

        q = X.shape[-2]
        num_dim = X.shape[-1]
        num_obs = len(self.model.train_inputs[0])
        # model_t.covar_module.base_kernel.lengthscale *= (max(1, num_obs) / (max(1, num_obs) + q)) ** (1/num_dim)
        model_t.covar_module.base_kernel.lengthscale *= ((1 + num_obs) / (1 + num_obs + q)) ** (1.0 / num_dim)
        var_t = model_t.posterior(self.X_samples, observation_noise=True).variance.squeeze()

        # var_t = var_t.mean(dim=-1)  # mean over X_samples
        # var_t = var_t.max(dim=-1).values # nicer designs than mean, but much slower

        # compromise: mu + sigma
        m = var_t.mean(dim=-1)
        s = var_t.std(dim=-1)

        # return -m
        return -(m + s)
