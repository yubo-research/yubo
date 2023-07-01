import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform
from IPython.core.debugger import set_trace
from torch import Tensor
from torch.quasirandom import SobolEngine

from sampling.pstar import PStar


class AcqPStar(MCAcquisitionFunction):
    """Model p*(x) as a Gaussian"""

    def __init__(self, model: Model, num_X_samples, num_Y_samples=None, beta=20, warm_start=None, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.num_Y_samples = num_Y_samples  # triggers joint sampling in inner loop; slower
        self.beta = beta
        self.sampler_pstar = SobolQMCNormalSampler(sample_shape=torch.Size([num_X_samples]))
        if num_Y_samples is not None:
            self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_Y_samples]))
        else:
            self.sampler = None

        X_0 = self.model.train_inputs[0].detach()
        num_obs = X_0.shape[0]
        num_dim = X_0.shape[-1]
        dtype = X_0.dtype

        sobol_engine = SobolEngine(num_dim, scramble=True)
        if num_obs == 0:
            self.X_samples = sobol_engine.draw(num_X_samples, dtype=dtype)
            self.prob_X_samples = torch.ones(num_X_samples) / num_X_samples
        else:
            self.X_samples, self.prob_X_samples, self.warm_start = self._draw_from_pstar(num_X_samples, warm_start)

    def _draw_from_pstar(self, num_X_samples, sigma_0):
        if sigma_0 is None:
            sigma_0 = 0.3
        mu = self._ts_max().cpu().detach().numpy()
        # Stolen from TurBO:
        # - idea of using kernel lengthscale for trust region aspect ratio
        # - the code that does it (next line, from turbo_1.py)
        cov_aspect = self.model.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        pstar = PStar(mu, cov_aspect, sigma_0=sigma_0)
        for _ in range(10):
            samples = pstar.ask(num_X_samples)
            X = torch.stack([torch.tensor(s.x) for s in samples])
            mvn = self.model.posterior(X, observation_noise=True)
            Y = self.sampler_pstar(mvn).squeeze(dim=-1)
            i_ts = torch.argmax(Y, dim=1)
            pstar.tell([samples[i] for i in i_ts])
            # print ("S:", pstar.sigma())
        samples = pstar.ask(num_X_samples)
        X_samples = torch.stack([torch.tensor(s.x) for s in samples])
        prob_X_samples = torch.tensor([s.p for s in samples])
        return X_samples, prob_X_samples, pstar.sigma()

    def _ts_max(self):
        X_0 = self.model.train_inputs[0].detach()
        if len(X_0) == 1:
            return X_0.flatten()
        Y = self.model.posterior(X_0).sample(torch.Size([1]))
        i = torch.argmax(Y.squeeze())
        return X_0[i]

    def _soft_entropy(self, model):
        mvn = model.posterior(self.X_samples, observation_noise=True)
        if self.num_Y_samples is None:
            mu = mvn.mean
            vr = mvn.variance
            p_max = torch.exp(self.beta * (mu + vr / 2))
            # p_max = p_max / p_max.sum(dim=-1).unsqueeze(dim=-1)
        else:
            Y = self.get_posterior_samples(mvn)
            p_max = torch.exp(self.beta * Y)
            p_max = p_max / p_max.sum(dim=-1).unsqueeze(dim=-1)
            p_max = p_max.mean(dim=0)
        p_max = p_max / p_max.sum(dim=-1).unsqueeze(dim=-1)
        # H = -(p_max * torch.log(p_max))
        H = -mu + vr / 2
        while len(H.shape) > 1:
            H = H.mean(dim=-1)
        return H

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        model_f = self.model.condition_on_observations(X=X, Y=self.model.posterior(X).mean)
        mvn_f = model_f.posterior(self.X_samples, observation_noise=True)
        if self.sampler is not None:
            Y = self.get_posterior_samples(mvn_f).squeeze(dim=-1)
            int_var = Y.var(dim=0).mean(dim=-1)
        else:
            int_var = mvn_f.variance.squeeze().mean(dim=-1)
        return -int_var
