import time

import numpy as np
import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform
# from IPython.core.debugger import set_trace
from torch import Tensor
from torch.quasirandom import SobolEngine

from sampling.fit_pstar import FitPStar


class AcqPStar(MCAcquisitionFunction):
    """Model p*(x) as a Gaussian"""

    def __init__(self, model: Model, num_X_samples, num_Y_samples=None, beta=20, use_soft_entropy=False, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.num_Y_samples = num_Y_samples  # triggers joint sampling in inner loop; slower
        self._beta = beta
        self._use_soft_entropy = use_soft_entropy
        self.sampler_pstar = SobolQMCNormalSampler(sample_shape=torch.Size([num_X_samples]))
        if num_Y_samples is not None:
            self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_Y_samples]))
        else:
            self.sampler = None

        X_0 = self.model.train_inputs[0].detach()
        num_obs = X_0.shape[0]
        self._num_dim = X_0.shape[-1]
        dtype = X_0.dtype

        sobol_engine = SobolEngine(self._num_dim, scramble=True)
        if num_obs == 0:
            self.X_samples = sobol_engine.draw(num_X_samples, dtype=dtype)
            self._prob_X_samples = torch.ones(num_X_samples) / num_X_samples
            self._unit_cov_0 = torch.ones(self._num_dim)
            self._dx2 = (self.X_samples - 0.5 * torch.ones(self._num_dim)) ** 2
        else:
            (self.X_samples, self._prob_X_samples, mu, self._unit_cov_0) = self._draw_from_pstar(num_X_samples)
            self._dx2 = (self.X_samples - mu) ** 2

        self._prob_X_samples = self._prob_X_samples.unsqueeze(0)
        self._dx2 = self._dx2.unsqueeze(0)
        self._unit_cov_0 = torch.tensor(self._unit_cov_0).unsqueeze(0).unsqueeze(0)

    def _unit_cov(self, model):
        # Stolen from TurBO:
        # - idea of using kernel lengthscale for trust region aspect ratio
        # - the code that does it (next line, from turbo_1.py)
        cov = self.model.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        assert np.all(cov > 0), cov

        cov = cov / cov.mean()
        det = np.prod(cov)
        unit_cov = cov / (det ** (1 / self._num_dim))
        assert np.abs(np.prod(unit_cov) - 1) < 1e-6
        return unit_cov

    def _draw_from_pstar(self, num_X_samples):
        mu = self._ts_max().cpu().detach().numpy()
        unit_cov = self._unit_cov(self.model)
        pstar = FitPStar(mu, unit_cov)
        t0 = time.time()
        while not pstar.converged():
            x, p = pstar.ask(num_X_samples)
            X = torch.tensor(x)
            mvn = self.model.posterior(X, observation_noise=True)
            Y = self.sampler_pstar(mvn).squeeze(dim=-1)
            i_ts = torch.argmax(Y, dim=1)
            pstar.tell(x[i_ts, :], p[i_ts])

        self.sigma = pstar.sigma()
        x, p = pstar.ask(num_X_samples, qmc=True)
        X_samples = torch.tensor(x)
        prob_X_samples = torch.tensor(p)
        self.fit_time = time.time() - t0
        return X_samples, prob_X_samples, mu, unit_cov

    def _ts_max(self):
        X_0 = self.model.train_inputs[0].detach()
        if len(X_0) == 1:
            return X_0.flatten()
        Y = self.model.posterior(X_0).sample(torch.Size([1]))
        i = torch.argmax(Y.squeeze())
        return X_0[i]

    """
    def _pdf_X_samples(self, sigma2):
        x = (self._dx2 / self._unit_cov_diag) / 2 / sigma2
        return torch.exp(-x.mean(dim=0)) / torch.sqrt(2 * torch.pi * sigma2)
    """

    def _soft_entropy(self, model):
        assert np.abs(self._unit_cov_0 - self._unit_cov(model)).max() < 1e-6, (self._unit_cov_0, self._unit_cov(model))
        mvn = model.posterior(self.X_samples, observation_noise=True)
        if self.num_Y_samples is None:
            mu = mvn.mean.squeeze(-1)
            vr = mvn.variance.squeeze(-1)
            p_max = torch.exp(self._beta * (mu + vr / 2))
            p_max = p_max / p_max.sum(dim=-1).unsqueeze(dim=-1)
        else:
            Y = self.get_posterior_samples(mvn)
            p_max = torch.exp(self._beta * Y).squeeze(-1)
            # rescale b/c exp() creates large scale
            p_max = p_max / p_max.sum(dim=-1).unsqueeze(dim=-1)
            p_max = p_max.mean(dim=0)

        weights = p_max / self._prob_X_samples
        weights = weights / weights.sum(dim=-1).unsqueeze(-1)
        weights = weights.unsqueeze(-1)
        # unit_cov doesn't seem to change upon conditioning,
        #  i.e., the relative length scales don't change,
        # so unit_cov_now = unit_cov_0
        sigma2 = ((weights * self._dx2 / self._unit_cov_0).sum(dim=-1) / weights.sum(dim=-1)).mean(dim=-1)

        # and det(cov_now) = sigma2 * det(unit_cov_now) = sigma2
        # since det(unit_cov_0) == 1 by construction.
        # The entropy of this multivariate Gaussian is:
        #   H = (num_dim/2)*(1 + ln(2pi)) + (1/2)[ln(sigma2)]
        # Since we're minimizing H, we don't care about the
        #  constants or the ln() (which is monotonic in its argument).
        return sigma2

    def _integrated_variance(self, model):
        mvn = model.posterior(self.X_samples, observation_noise=True)
        if self.sampler is not None:
            Y = self.get_posterior_samples(mvn).squeeze(dim=-1)
            vr = Y.var(dim=0)
        else:
            vr = mvn.variance.squeeze()

        return vr.mean(dim=-1)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        model_f = self.model.condition_on_observations(X=X, Y=self.model.posterior(X).mean)

        if self._use_soft_entropy:
            return -self._soft_entropy(model_f)
        return -self._integrated_variance(model_f)
