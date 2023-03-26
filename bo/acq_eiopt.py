from typing import Optional

import torch
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
)
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform

# from IPython.core.debugger import set_trace
from torch import Tensor
from torch.quasirandom import SobolEngine


# TODO: optional MC joint samples of Y in fantasy model
#        to capture covariance/cross terms
class AcqEIOpt(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        num_X_samples: int = 256,
        num_ts_samples: int = None,
        num_Y_samples: int = None,
        b_noisy: bool = False,
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
        # dtype = X_0.dtype

        self.X_samples = self._sample_x(model, num_dim, num_X_samples, num_ts_samples, b_noisy)
        # TODO: weights, too?
        self.weights = torch.ones(size=(len(self.X_samples),))
        self.weights = self.weights / self.weights.sum()

    def _sample_x(self, model, num_dim, num_X_samples, num_ts_samples, b_noisy):
        X_samples = []
        sobol_engine = SobolEngine(num_dim, scramble=True)
        for _ in range(num_X_samples):
            x = sobol_engine.draw(num_ts_samples)
            if b_noisy:
                gp = self._get_ts_model(model)
            else:
                gp = model
            y = gp.posterior(x).sample().squeeze(0).squeeze(-1).detach()
            i = torch.argmax(y + 1e-9 * torch.randn(size=y.shape))
            X_samples.append(x[i])
        return torch.stack(X_samples)

    def _xxx_calc_weights(self, model, X_samples, num_p_best_samples, b_noisy):
        if b_noisy:
            gp_ts = self._thompson_sample_models(model, num_p_best_samples)
            y = gp_ts.posterior(X_samples).sample().squeeze(0).squeeze(-1).detach()
        else:
            y = model.posterior(X_samples.repeat(num_p_best_samples, 1, 1)).sample().squeeze(0).squeeze(-1).detach()

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

    def _thompson_sample_models(self, model, num_p_best_samples):
        x = model.train_inputs[0].detach()
        x = x.repeat(num_p_best_samples, 1, 1)
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
