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

    def __init__(self, model: Model, num_X_samples: int = 128, k_p_max: int = 10, num_mcmc: int = 10, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.num_X_samples = num_X_samples
        self.num_px_samples = k_p_max * num_X_samples
        self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_px_samples]))

        X_samples = self._sample_X(num_X_samples, num_mcmc)
        assert len(X_samples) == num_X_samples, len(X_samples)
        self.X_samples = X_samples

        with torch.no_grad():
            self.p_0 = ((1.0 / self.num_X_samples) / 10.0).detach()
            self.p_max = self._calc_p_max(self.model, self.X_samples).detach()
            self.weights = self.p_max.clone()
            self.p_max_reg = self.p_0 + self._calc_p_max(self.model, self.X_samples).detach()

    def _sample_X(self, num_X_samples, num_mcmc):
        x_max = self._find_max(self.model).detach()
        X_samples = torch.tile(x_max, (num_X_samples, 1))

        eps_mc = 0.1
        # burn in
        for _ in range(num_mcmc):
            X_samples = self._mcmc(X_samples, eps=eps_mc)
        # collect paths
        for _ in range(num_mcmc):
            X_samples = self._mcmc(X_samples, eps=eps_mc)

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

    def _mcmc(self, X, eps):
        with torch.no_grad():
            X_new = X + eps * torch.randn(size=X.shape)
            i = np.where((X_new < 0) | (X_new > 1))[0]
            X_new[i] = torch.rand(size=(len(i), X.shape[-1])).type(X.dtype)
            assert torch.all((X_new >= 0) & (X_new <= 1)), X_new
            X_both = torch.cat((X, X_new), axis=0)
            p_all = 1e-9 + self._calc_p_max(self.model, X_both)[:, None].mean(axis=1)
            p = p_all[: len(X)]
            p_new = p_all[len(X) :]

            a = p_new / p
            u = torch.rand(size=(len(X),))
            i = u <= a
            X[i] = X_new[i]
        return X

    def _calc_p_max(self, model, X):
        posterior = model.posterior(X, posterior_transform=self.posterior_transform, observation_noise=True)
        Y = posterior.sample(torch.Size([self.num_px_samples])).squeeze(dim=-1)  # num_Y_samples x b x len(X)
        return self._calc_p_max_from_Y(Y)

    def _calc_p_max_from_Y(self, Y):
        beta = 12
        sm = torch.exp(beta * Y)
        sm = sm / sm.sum(dim=-1).unsqueeze(-1)
        p_max = sm.mean(dim=0)
        assert np.abs(p_max.sum() - 1) < 1e-4, p_max
        return p_max

    def _soft_p_max(self, model):
        posterior = model.posterior(self.X_samples, observation_noise=True)
        Y = self.get_posterior_samples(posterior).squeeze(dim=-1)  # num_Y_samples x b x num_X_samples
        beta = 12
        sm = torch.exp(beta * Y)
        sm = sm / sm.sum(dim=-1).unsqueeze(-1)
        p_max = sm.mean(dim=0)
        return p_max

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self.to(device=X.device)

        mvn = self.model.posterior(X)
        model_t = self.model.condition_on_observations(X=X, Y=mvn.mean)  # TODO: noise=observation_noise
        p_max_t = self._soft_p_max(model_t)

        # Regularize to protect against small p_max's.
        p_max_t_reg = self.p_0 + p_max_t
        w_imp_reg = p_max_t_reg / self.p_max_reg
        w_imp_reg = w_imp_reg / w_imp_reg.sum()

        # p_max is calculated up to a normalizing constant.
        # That constant is different for every model_t.
        # So how could we compare different model_t's?

        # Approximate the negative entropy: E[ln p_max_t(x)]
        # return (w_imp_reg * torch.log(p_max_reg)).sum(dim=-1))
        return (w_imp_reg * torch.log(w_imp_reg)).sum(dim=-1)
