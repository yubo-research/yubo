from typing import Optional

import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform

# from IPython.core.debugger import set_trace
from torch import Tensor
from torch.quasirandom import SobolEngine


class AcqIDOpt(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        bounds: torch.Tensor,
        X_max: torch.Tensor = None,
        num_X_samples: int = 256,
        num_Y_samples: int = 16,
        sampler: Optional[MCSampler] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs
    ) -> None:
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_Y_samples]))
        self.sampler = sampler

        X_0 = self.model.train_inputs[0]
        num_dim = X_0.shape[-1]
        sobol_engine = SobolEngine(num_dim, scramble=True)

        X_samples = sobol_engine.draw(num_X_samples, dtype=X_0.dtype)
        X_samples = bounds[0] + (bounds[1] - bounds[0]) * X_samples
        X_samples = X_samples.view(num_X_samples, num_dim).to(device=X_0.device)
        if X_max is not None:
            X_max = torch.atleast_2d(X_max)
            X_all = torch.concatenate((X_max, X_samples), axis=0)  # want a joint sample
            self.use_x_max = True
        else:
            X_all = X_samples
            self.use_x_max = False
        self.register_buffer("X_all", X_all)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        posterior = self.model.posterior(X, posterior_transform=self.posterior_transform)  # predictive posterior

        num_batches = X.shape[0]
        af = []

        for i_batch in range(num_batches):
            Y = posterior.mean[i_batch, :]  # q
            model_next = self.model.condition_on_observations(X=X[i_batch, ::], Y=Y)  # q x d

            post_posterior = model_next.posterior(self.X_all, posterior_transform=self.posterior_transform)
            Y_samples = self.get_posterior_samples(post_posterior).squeeze()  # num_Y_samples x (1 + num_X_samples)
            if self.use_x_max:
                Y_max = Y_samples[:, :1]
                Y_samples = Y_samples[:, 1:]
                delta = Y_max - Y_samples
            else:
                delta = Y_samples
            var = delta.var(axis=0)  # num_X_samples
            af.append(-var.mean())

        return torch.stack(af)
