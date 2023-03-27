from typing import Optional

import gpytorch
import torch
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
)
from botorch.models.model import Model
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
        b_adaptive_x_sampling: bool = True,
        num_ts_samples: int = 1000,
        num_Y_samples: int = None,
        b_joint_sampling: bool = False,
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

        if b_adaptive_x_sampling:
            self.X_samples = self._adaptive_samples(model, num_dim, num_X_samples, num_ts_samples, b_joint_sampling)
        else:
            self.X_samples = self._sobol_samples(model, num_dim, num_X_samples)
        self.weights = self._calc_weights(model, self.X_samples, num_ts_samples, b_joint_sampling)

    def _sobol_samples(self, model, num_dim, num_X_samples):
        sobol_engine = SobolEngine(num_dim, scramble=True)
        return sobol_engine.draw(num_X_samples)

    def _adaptive_samples(self, model, num_dim, num_X_samples, num_ts_samples, b_joint_sampling):
        X_samples = []
        for _ in range(num_X_samples):
            sobol_engine = SobolEngine(num_dim, scramble=True)
            X = sobol_engine.draw(num_ts_samples)
            Y = self._sample_y(model, X, b_joint_sampling) + 1e-9 * torch.randn(size=(len(X),))
            X_samples.append(X[torch.argmax(Y)])
        return torch.stack(X_samples)

    def _sample_y(self, model, x, b_joint_sampling):
        pred = model.likelihood(model(x))
        if b_joint_sampling:
            with torch.no_grad():
                with gpytorch.settings.ciq_samples(True), gpytorch.settings.num_contour_quadrature(10), gpytorch.settings.minres_tolerance(1e-4):
                    y = pred.sample().squeeze(0).squeeze(-1).detach()
        else:
            y = pred.mean + pred.stddev * (torch.randn(size=x.shape[:-1]))
        return y

    def _calc_weights(self, model, X_samples, num_ts_samples, b_joint_sampling):
        X = X_samples.repeat(num_ts_samples, 1, 1)
        y = self._sample_y(model, X, b_joint_sampling)

        i_best = torch.argmax(y, dim=-1)
        i, counts = torch.unique(i_best, return_counts=True)
        p_best = torch.zeros(size=(len(X_samples),)).type(y.dtype)
        p_best[i] = counts.type(y.dtype)
        return p_best / p_best.sum()

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
