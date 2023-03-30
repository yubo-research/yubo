from typing import Optional

import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform

# from IPython.core.debugger import set_trace
from torch import Tensor
from torch.quasirandom import SobolEngine


class AcqIntegrated(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        bounds: Tensor,
        num_X_samples: int = 256,
        num_Y_samples: int = 64,
        X_special: Tensor = None,
        sampler: Optional[MCSampler] = None,
        **kwargs
    ) -> None:
        super().__init__(model=model, **kwargs)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_Y_samples]))
        self.sampler = sampler

        X_0 = self.model.train_inputs[0]
        num_dim = X_0.shape[-1]
        sobol_engine = SobolEngine(num_dim, scramble=True)
        X_samples = sobol_engine.draw(num_X_samples, dtype=X_0.dtype)
        X_samples = bounds[0] + (bounds[1] - bounds[0]) * X_samples
        X_samples = X_samples.view(num_X_samples, num_dim).to(device=X_0.device)

        if X_special is not None:
            X_special = torch.atleast_2d(X_special)
            self.n_special = len(X_special)
            X_samples = torch.concatenate((X_special, X_samples), axis=0)  # want a joint sample
        else:
            self.n_special = 0

        self.register_buffer("X_samples", X_samples)

    def fantasy_observation(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X, observation_noise=True)  # predictive posterior
        return posterior.mean  # (b) x q

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        Y_obs = self.fantasy_observation(X)  # (b) x q

        num_batches = X.shape[0]
        af = []

        for i_batch in range(num_batches):
            Y = Y_obs[i_batch, :]  # q
            # model_next is conditioned on having taken *all* q measurements.
            model_next = self.model.condition_on_observations(X=X[i_batch, ::], Y=Y)  # X[] is q x d
            next_posterior = model_next.posterior(self.X_samples)
            Y_samples = self.get_posterior_samples(next_posterior).squeeze()  # num_Y_samples x num_X_samples

            if self.n_special > 0:
                Y_special = Y_samples[:, : self.n_special]
                Y_samples = Y_samples[:, self.n_special :]
            else:
                Y_special = None

            integrand = self.integrand(Y_samples, Y_special)
            af.append(-integrand.mean())

        return torch.stack(af)
