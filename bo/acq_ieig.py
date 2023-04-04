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


class AcqIEIG(MCAcquisitionFunction):
    """Integrated Expected Information Gain
    Maximize the integral (over X) of the EIG of the distribution
     of the optimal x: p_*(x).

     - Follows optimal Bayesian experiment design (OED) by maximizing the EIG.
     - Measure T Thompson samples with q arms
     - Similar to integrated optimality / elastic integrated optimality
    """

    def __init__(
        self,
        model: Model,
        num_X_samples: int = 64,
        num_px_samples = 4096,
        num_Y_samples: int = 1024,
        num_noisy_maxes: int = 1,
        joint_sampling: bool = True,
        **kwargs
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_Y_samples]))
        self.num_px_samples = num_px_samples
        self.joint_sampling = joint_sampling

        # SLOW: num_noisy_maxes was 10
        X_samples = self._sample_X(num_noisy_maxes, num_X_samples)
        assert len(X_samples) >= num_X_samples, len(X_samples)
        i = np.random.choice(np.arange(len(X_samples)), size=(num_X_samples,), replace=False)
        self.X_samples = X_samples[i]

        # Diagnostics
        with torch.no_grad():
            self.p_max = self._calc_p_max(self.model, self.X_samples)
            self.weights = self.p_max.clone()
            if True:
                # some points never move b/c they and
                #  their proposals have zero probability
                th = 0.1 / len(self.X_samples)
                self.weights[self.weights < th] = 0.0
                self.weights[self.weights >= th] = 1.0
                self.weights = self.weights / self.weights.sum()

    def _sample_X(self, num_noisy_maxes, num_X_samples):
        X = self.model.train_inputs[0]

        if len(X) == 0:
            return self._sobol_samples(num_X_samples)
        else:
            no2 = num_X_samples

            models = [self._get_noisy_model() for _ in range(num_noisy_maxes)]
            x_max = []
            for model in models:
                x_max.append(self._find_max(self.model).detach())
            x_max = torch.cat(x_max, axis=0)
            n = int(no2 / len(x_max) + 1)
            # X_samples = torch.cat((X_samples, torch.tile(x_max, (n, 1))), axis=0)
            X_samples = torch.tile(x_max, (n, 1))
            assert len(X_samples) >= num_X_samples, (len(X_samples), num_X_samples, no2)

            # burn in
            for _ in range(10):
                self._mcmc(models, X_samples, eps=0.1)

            # collect paths
            X_all = []
            for _ in range(10):
                X_samples = self._mcmc(models, X_samples, eps=0.1)
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

    def _mcmc(self, models, X, eps):
        # TODO: keep the whole path (after warm-up)
        # TODO: NUTS
        with torch.no_grad():
            X_new = X + eps * torch.randn(size=X.shape)
            i = np.where((X_new < 0) | (X_new > 1))[0]
            X_new[i] = torch.rand(size=(len(i), X.shape[-1])).type(X.dtype)
            assert torch.all((X_new >= 0) & (X_new <= 1)), X_new
            X_both = torch.cat((X, X_new), axis=0)
            p_all = 1e-9 + torch.cat([self._calc_p_max(m, X_both)[:, None] for m in models], axis=1).mean(axis=1)
            # p_all = 1e-9 + self._calc_p_max(self.model, X_both)[:, None].mean(axis=1)
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

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        Y = self.model.posterior(X).mean  # q x 1
        model_t = self.model.condition_on_observations(X=X, Y=Y)

        # No fantazing b/c variance is independent of Y.
        posterior_t = model_t.posterior(self.X_samples, posterior_transform=self.posterior_transform, observation_noise=True)

        if self.joint_sampling:
            Y = self.get_posterior_samples(posterior_t).squeeze(dim=-1)
            var_t = Y.var(dim=0)
        else:
            # skip joint sampling for speed (?)
            var_t = posterior_t.variance.squeeze()

        return -(self.weights * var_t).sum(dim=-1)
