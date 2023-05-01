import numpy as np
import torch
from botorch.acquisition import PosteriorMean
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
)
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform

# from IPython.core.debugger import set_trace
from torch import Tensor
from torch.quasirandom import SobolEngine


class AcqDES(MCAcquisitionFunction):
    """Direct Entropy Search"""

    def __init__(
        self,
        model: Model,
        num_X_samples: int = 128,
        num_px_samples: int = 128,
        num_mcmc: int = 10,
        p_all_type: str = "all",
        num_fantasies: int = 4,
        num_Y_samples: int = 32,
        num_noisy_maxes: int = 3,
        fantasies_only=False,
        **kwargs
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_Y_samples]))
        if num_fantasies > 0:
            self.sampler_fantasies = SobolQMCNormalSampler(sample_shape=torch.Size([num_fantasies]))
        self.num_fantasies = num_fantasies
        self.num_Y_num_f = num_Y_samples * num_fantasies
        self.num_Y_samples = num_Y_samples
        self.num_px_samples = num_px_samples
        self._fantasies_only = fantasies_only
        
        if len(self.model.train_inputs[0]) == 0:
            self.X_samples = self._sobol_samples(num_X_samples)
            self._p_max = torch.ones(len(self.X_samples))
            self._p_max = self._p_max / self._p_max.sum()
        else:
            X_samples = self._sample_X(num_noisy_maxes, num_X_samples, num_mcmc, p_all_type)

            assert len(X_samples) >= num_X_samples, len(X_samples)
            if len(X_samples) != num_X_samples:
                i = np.random.choice(np.arange(len(X_samples)), size=(num_X_samples,), replace=False)
                X_samples = X_samples[i]

            self.X_samples = X_samples
            self._p_max = self._calc_p_max(self.model, self.X_samples)
            print ("P:", self._p_max.max() / self._p_max.min())

    def thompson_sample(self, q):
        i = np.random.choice(np.arange(len(self.X_samples)), size=(q,))
        return torch.atleast_2d(self.X_samples[i])

    def _sample_X(self, num_noisy_maxes, num_X_samples, num_mcmc, p_all_type):
        if num_noisy_maxes == 0:
            models = [self.model]
        else:
            models = [self._get_noisy_model() for _ in range(num_noisy_maxes)]
        x_max = []
        for model in models:
            x = self._find_max(model).detach()
            x_max.append(x)
        x_max = torch.cat(x_max, axis=0)
        no2 = num_X_samples
        n = int(no2 / len(x_max) + 1)

        X_samples = torch.tile(x_max, (n, 1))
        assert len(X_samples) >= num_X_samples, (len(X_samples), num_X_samples, no2)

        # burn in
        for _ in range(num_mcmc):
            X_samples = self._mcmc(models, X_samples, eps=0.1, p_all_type=p_all_type)

        # collect paths
        X_all = []
        for _ in range(num_mcmc):
            X_samples = self._mcmc(models, X_samples, eps=0.1, p_all_type=p_all_type)
            X_all.append(X_samples)
        X_all = torch.cat(X_all, axis=0)
        return X_all

    def _get_noisy_model(self):
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

    def _mcmc(self, models, X, eps, p_all_type):
        with torch.no_grad():
            X_new = X + eps * torch.randn(size=X.shape)
            i = np.where((X_new < 0) | (X_new > 1))[0]
            X_new[i] = torch.rand(size=(len(i), X.shape[-1])).type(X.dtype)
            assert torch.all((X_new >= 0) & (X_new <= 1)), X_new
            X_both = torch.cat((X, X_new), axis=0)
            if p_all_type == "all":
                p_all = 1e-9 + torch.cat([self._calc_p_max(m, X_both)[:, None] for m in models], axis=1).mean(axis=1)
            elif p_all_type == "random":
                p_all = 1e-9 + self._calc_p_max(np.random.choice(models), X_both)[:, None].mean(axis=1)
            elif p_all_type == "self":
                p_all = 1e-9 + self._calc_p_max(self.model, X_both)[:, None].mean(axis=1)
            else:
                assert False, p_all_type
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

    def _calc_p_max_from_Y(self, Y, across_batches=False):
        beta = 12
        sm = torch.exp(beta * Y)
        if across_batches:
            # Y ~ num_Y_samples x num_fantasies x b x num_X_samples
            if self._fantasies_only:
                norm = sm.sum(dim=-1).sum(dim=0)
                norm = norm.unsqueeze(-1).unsqueeze(0)
            else:
                norm = sm.sum(dim=-1).sum(dim=-1).sum(dim=0)
                norm = norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        else:
            norm = sm.sum(dim=-1).unsqueeze(-1)
        sm = sm / norm

        p_max = sm.mean(dim=0)
        if not across_batches:
            assert np.abs(p_max.sum() - 1) < 1e-4, (p_max.sum(), p_max.shape, Y.shape)
        return p_max

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        model_f = self.model.fantasize(X=X, sampler=self.sampler_fantasies, observation_noise=True)
        mvn_f = model_f.posterior(self.X_samples, observation_noise=True)
        Y_f = self.get_posterior_samples(mvn_f).squeeze(dim=-1)  # num_Y_samples x num_fantasies x b x num_X_samples
        assert Y_f.shape[0] == self.num_Y_samples, Y_f.shape
        p_max = self._calc_p_max_from_Y(Y_f, across_batches=True)  # num_fantasies x b x num_X_samples
        H = -((p_max / self._p_max) * torch.log(p_max)).mean(dim=-1).mean(dim=0)
        # H = (torch.log(p_max)).mean(dim=-1).mean(dim=0)

        return -H
