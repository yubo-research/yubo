from typing import Optional

import gpytorch
import torch
from botorch.acquisition import PosteriorMean
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
)
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform

# from IPython.core.debugger import set_trace
from torch import Tensor
from torch.quasirandom import SobolEngine


class AcqEIOpt(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        num_X_samples: int = 256,
        num_ts_samples: int = 1000,
        num_Y_samples: int = None,
        b_joint_weights: bool = False,
        sampler: Optional[MCSampler] = None,
        **kwargs
    ) -> None:
        super().__init__(model=model, **kwargs)
        if num_Y_samples:
            if sampler is None:
                sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_Y_samples]))
            self.sampler = sampler
        self.num_Y_samples = num_Y_samples

        X_0 = self.model.train_inputs[0]
        num_dim = X_0.shape[-1]

        self.X_samples = self._mk_x_samples(model, num_dim, num_X_samples)
        self.weights = self._calc_weights(model, self.X_samples, num_ts_samples, b_joint_weights)

    def _mk_x_samples(self, model, num_dim, num_X_samples):
        sobol_engine = SobolEngine(num_dim, scramble=True)
        X_samples = sobol_engine.draw(num_X_samples)
        if len(self.model.train_inputs[0]) > 0:
            X_samples = torch.cat((X_samples, self._find_max(self._get_ts_model(model))), axis=0)
        return X_samples

    def _find_max(self, gp):
        X_0 = self.model.train_inputs[0]
        num_dim = X_0.shape[-1]
        x_cand, _ = optimize_acqf(
            acq_function=PosteriorMean(model=gp),
            bounds=torch.tensor([[0.0] * num_dim, [1.0] * num_dim], device=X_0.device, dtype=X_0.dtype),
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 10, "maxiter": 200},
        )
        return x_cand

    def _calc_weights(self, model, X_samples, num_ts_samples, b_joint_weights):
        X = X_samples.repeat(num_ts_samples, 1, 1)

        pred = model.likelihood(model(X))
        if b_joint_weights:
            with torch.no_grad():
                with gpytorch.settings.ciq_samples(True), gpytorch.settings.num_contour_quadrature(10), gpytorch.settings.minres_tolerance(1e-4):
                    y = pred.sample().squeeze(0).squeeze(-1).detach()
        else:
            y = pred.mean + pred.stddev * (torch.randn(size=X.shape[:-1]))

        i_best = torch.argmax(y, dim=-1)
        i, counts = torch.unique(i_best, return_counts=True)
        p_best = torch.zeros(size=(len(X_samples),)).type(y.dtype)
        p_best[i] = counts.type(y.dtype)
        return p_best / p_best.sum()

    def _get_ts_model(self, model):
        x = model.train_inputs[0].detach()
        y = model.posterior(x, observation_noise=True).sample().squeeze(0).detach()

        model_ts = SingleTaskGP(x, y, model.likelihood)
        model_ts.initialize(**dict(model.named_parameters()))
        model_ts.eval()
        return model_ts

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        Y = self.model.posterior(X, observation_noise=True).mean  # b x q x 1
        model_t = self.model.condition_on_observations(X=X, Y=Y)
        posterior_t = model_t.posterior(self.X_samples, observation_noise=True)
        if self.num_Y_samples:
            var_t = self.get_posterior_samples(posterior_t).squeeze().var(dim=0)
        else:
            var_t = posterior_t.variance.squeeze()

        mean_var_t = (self.weights * var_t).sum(dim=-1)  # mean over X_samples

        return -mean_var_t
