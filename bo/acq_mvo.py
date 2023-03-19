from typing import Optional

import torch
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
)
from botorch.generation import MaxPosteriorSampling
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform
from torch import Tensor
from torch.quasirandom import SobolEngine

# from IPython.core.debugger import set_trace


class AcqMVO(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        num_thompson_candidates: int = 1000,
        num_X_samples: int = 256,
        num_Y_samples: int = 256,
        sampler: Optional[MCSampler] = None,
        **kwargs
    ) -> None:
        super().__init__(model=model, **kwargs)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_Y_samples]))
        self.sampler = sampler

        X_0 = self.model.train_inputs[0]
        num_dim = X_0.shape[-1]
        dtype = X_0.dtype
        device = X_0.device

        sobol_engine = SobolEngine(num_dim, scramble=True)
        X_candidates = sobol_engine.draw(num_thompson_candidates, dtype=dtype).to(device)
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=True)
        self.X_samples = thompson_sampling(X_candidates, num_samples=num_X_samples).type(dtype).to(device)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        # condition fantasy model on measurement of arms
        Y = self.model.posterior(X).mean  # b x q x 1
        model_t = self.model.condition_on_observations(X=X, Y=Y)

        posterior_t = model_t.posterior(self.X_samples, observation_noise=True)

        # num_Y_samples x b x num_X_samples x 1
        Y_samples = self.get_posterior_samples(posterior_t)

        # var over Y samples
        # mean over X_samples
        var_t = Y_samples.var(dim=0).squeeze(-1).mean(dim=-1)
        return -var_t  # b
