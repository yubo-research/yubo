import numpy as np
import torch
from botorch.acquisition import PosteriorMean
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
)
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform
from torch import Tensor
from torch.quasirandom import SobolEngine


class AcqAEIG(MCAcquisitionFunction):
    """Approximate Expected Information Gain
    Maximize the approximate EIG of the distribution
     of the optimal x, p_max(x).

    - Follows optimal Bayesian experiment design (OED) by maximizing the EIG.
    - No initialization step period.
    """

    def __init__(self, model: Model, num_X_samples: int = 128, num_Y_samples: int = 10, num_mcmc: int = 10, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.num_X_samples = num_X_samples
        self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_Y_samples]))

        # X_samples = self._sobol_samples(num_X_samples)
        X_samples = self._sample_X(num_X_samples, num_mcmc).detach()

        assert len(X_samples) == num_X_samples, len(X_samples)
        self.X_samples = X_samples

    def _sample_X(self, num_X_samples, num_mcmc):
        X_samples = self._sobol_samples(num_X_samples)  # // 2)
        if len(X_samples) < num_X_samples:
            # Needed this in high dimensions?
            x_max = self._find_max(self.model)
            X_samples = torch.cat((X_samples, torch.tile(x_max, (num_X_samples - len(X_samples), 1))), axis=0).detach()

        with torch.no_grad():
            for eps_mc in [0.3, 0.1, 0.03]:
                for _ in range(num_mcmc):
                    X_samples = self._tmc(X_samples, eps=eps_mc)

            if True:
                # clean up noise
                posterior = self.model.posterior(X_samples, posterior_transform=self.posterior_transform, observation_noise=True)
                Y = posterior.sample(torch.Size([100])).squeeze(dim=-1)  # num_Y_samples x b x len(X)
                i_best = torch.argmax(Y, dim=-1)
                i = torch.unique(i_best)
                X_samples = X_samples[i]
                i = np.random.choice(np.arange(len(X_samples)), size=(num_X_samples,), replace=True)
                X_samples = X_samples[i]
        return X_samples

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

    def _tmc(self, X, eps):
        X_eps = X + eps * torch.randn(size=X.shape)
        i = np.where((X_eps < 0) | (X_eps > 1))[0]
        X_eps[i] = torch.rand(size=(len(i), X.shape[-1])).type(X.dtype)
        assert torch.all((X_eps >= 0) & (X_eps <= 1)), X_eps
        X_both = torch.cat((X, X_eps), axis=0)

        posterior = self.model.posterior(X_both, observation_noise=True)
        Y_both = posterior.sample(torch.Size([1])).squeeze()  # len(X)
        Y = Y_both[: len(X)]
        Y_eps = Y_both[-len(X_eps) :]

        X_new = X.clone()
        i = Y_eps > Y
        X_new[i] = X_eps[i]
        return X_new

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self.to(device=X.device)

        mvn = self.model.posterior(X, observation_noise=True)
        Y = self.get_posterior_samples(mvn).squeeze(dim=-1)  # num_Y_samples x b x q

        model_t = self.model.condition_on_observations(X=X, Y=mvn.mean)  # TODO: noise=observation_noise
        mvn_t = model_t.posterior(self.X_samples, observation_noise=True)
        Y_t = self.get_posterior_samples(mvn_t).squeeze(dim=-1)  # num_Y_samples x b x num_X_samples

        if True:
            H = torch.log(mvn.stddev).mean(dim=-1)
            H_t = torch.log(mvn_t.stddev).mean(dim=-1)
        else:
            H = torch.log(Y.std(dim=0)).mean(dim=-1)
            H_t = torch.log(Y_t.std(dim=0)).mean(dim=-1)

        return H - H_t
