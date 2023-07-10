import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction, qSimpleRegret
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform

# from IPython.core.debugger import set_trace
from torch import Tensor
from torch.quasirandom import SobolEngine


class AcqSRMV(MCAcquisitionFunction):
    def __init__(self, model: Model, num_X_samples=64, num_Y_samples=256, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self._num_X_samples = num_X_samples
        self.num_Y_samples = num_Y_samples

        # self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_Y_samples]))

        X_0 = self.model.train_inputs[0].detach()
        self._num_obs = X_0.shape[0]
        self._num_dim = X_0.shape[-1]
        self._dtype = X_0.dtype

        sobol_engine = SobolEngine(self._num_dim, scramble=True)
        self.X_samples = sobol_engine.draw(num_X_samples, dtype=self._dtype)
        self._sr = qSimpleRegret(model=self.model, sampler=SobolQMCNormalSampler(sample_shape=torch.Size([num_Y_samples])))
        if len(self.model.train_targets) > 0:
            self._Y_max = self.model.train_targets.max()
        else:
            self._Y_max = None

    def _max(self, model, X):
        mvn = model.posterior(X, observation_noise=True)
        Y = self.get_posterior_samples(mvn).squeeze(dim=-1)
        Y_max = Y.max(dim=0)
        Y_maxmax = Y_max.values.max(dim=1)
        i = Y_max.indices[:, Y_maxmax.indices].diag()
        return torch.diagonal(Y[i, :, :], dim1=0, dim2=1).T

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        q = X.shape[-2]
        assert len(self.X_samples) >= 10 * q, "You should use num_X_samples >= 10*q"
        num_dim = X.shape[-1]
        num_obs = len(self.model.train_inputs[0])

        # Y_obs = self._max(self.model, X)
        mvn = self.model.posterior(X)
        Y_obs = mvn.mean.squeeze(-1)
        # if self._Y_max is not None:
        #    Y_obs = torch.maximum(torch.tensor(0.), Y_obs - self._Y_max)
        # if q == 1:
        #    Y_obs = Y_obs.unsqueeze(-1)
        # set_trace()

        # TODO: Y_obs?
        # model_f = self.model.condition_on_observations(X=X, Y=Y_obs)
        model_f = self.model.condition_on_observations(X=X, Y=self.model.posterior(X).mean)
        model_f.covar_module.base_kernel.lengthscale *= ((1 + num_obs) / (1 + num_obs + q)) ** (1.0 / num_dim)

        var_f = model_f.posterior(self.X_samples, observation_noise=True).variance.squeeze()
        m = var_f.mean(dim=-1)
        s = var_f.std(dim=-1)
        sd = torch.sqrt((m + s) / 2)

        return Y_obs.max(dim=-1).values - sd
