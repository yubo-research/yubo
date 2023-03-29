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


class AcqTIOpt(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        num_X_samples_per_dim: int = 4,
        num_ts_samples: int = 1024,
        num_Y_samples: int = None,
            b_concentrate: bool = False,
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

        X = model.train_inputs[0]
        num_dim = X.shape[-1]
        num_X_samples = num_X_samples_per_dim * num_dim

        self.X_samples = self._sobol_samples(num_X_samples)
        self.X_samples = torch.cat((self.X_samples, self._adaptive_samples(int(num_X_samples / 2), num_ts_samples, b_joint_sampling)), axis=0)
        self.X_samples = torch.cat((self.X_samples, self._noisy_maxes(max(1, int(num_X_samples / 4)))), axis=0)
        self.weights = self._calc_weights(self.X_samples, num_ts_samples, b_joint_sampling, b_concentrate)

    def _noisy_maxes(self, num_X_samples):
        X_0 = self.model.train_inputs[0]
        X_samples = []
        for _ in range(num_X_samples):
            X_samples.append(self._find_max(self._get_ts_model()))
        return torch.cat(X_samples, axis=0).to(X_0.device).type(X_0.dtype)

    def _get_ts_model(self):
        X = self.model.train_inputs[0].detach()
        if len(X) == 0:
            return self.model

        Y = self.model.posterior(X, observation_noise=True).sample().squeeze(0).detach()

        model_ts = SingleTaskGP(X, Y, self.model.likelihood)
        model_ts.initialize(**dict(self.model.named_parameters()))
        model_ts.eval()
        return model_ts

    def _find_max(self, model):
        X = model.train_inputs[0]
        num_dim = X.shape[-1]

        x_cand, _ = optimize_acqf(
            acq_function=PosteriorMean(model),
            bounds=torch.tensor([[0.0] * num_dim, [1.0] * num_dim], device=X.device, dtype=X.dtype),
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 10, "maxiter": 200},
        )
        return x_cand

    def _sobol_samples(self, num_X_samples):
        X_0 = self.model.train_inputs[0]
        num_dim = X_0.shape[-1]
        sobol_engine = SobolEngine(num_dim, scramble=True)
        return sobol_engine.draw(num_X_samples).to(X_0.device).type(X_0.dtype)

    def _adaptive_samples(self, num_X_samples, num_ts_samples, b_joint_sampling):
        X_0 = self.model.train_inputs[0]
        num_dim = X_0.shape[-1]
        X_samples = []
        for _ in range(num_X_samples):
            sobol_engine = SobolEngine(num_dim, scramble=True)
            X = sobol_engine.draw(num_ts_samples)
            Y = self._sample_y(self.model, X, b_joint_sampling) + 1e-9 * torch.randn(size=(len(X),))
            X_samples.append(X[torch.argmax(Y)])
        return torch.stack(X_samples).to(X_0.device).type(X_0.dtype)

    def _sample_y(self, model, x, b_joint_sampling):
        pred = model.likelihood(model(x))
        if b_joint_sampling:
            with torch.no_grad():
                with gpytorch.settings.ciq_samples(True), gpytorch.settings.num_contour_quadrature(10), gpytorch.settings.minres_tolerance(1e-4):
                    y = pred.sample().squeeze(0).squeeze(-1).detach()
        else:
            y = pred.mean + pred.stddev * (torch.randn(size=x.shape[:-1]))
        return y

    def _calc_weights(self, X_samples, num_ts_samples, b_joint_sampling, b_concentrate):
        X = X_samples.repeat(num_ts_samples, 1, 1)
        y = self._sample_y(self.model, X, b_joint_sampling)

        i_best = torch.argmax(y, dim=-1)
        i, counts = torch.unique(i_best, return_counts=True)
        p_best = torch.zeros(size=(len(X_samples),))
        p_best[i] = counts.type(p_best.dtype)

        X_0 = self.model.train_inputs[0]
        num_dim = X_0.shape[-1]
        p_best = p_best.type(torch.float64)
        p_best = p_best / p_best.sum()
        if b_concentrate:
            p_best = p_best**num_dim
            p_best = p_best / p_best.sum()
        return p_best.to(X_0.device).type(X_0.dtype)

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
