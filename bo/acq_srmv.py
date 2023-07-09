import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
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

        self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_Y_samples]))
        self.joint_variance = True

        X_0 = self.model.train_inputs[0].detach()
        self._num_obs = X_0.shape[0]
        self._num_dim = X_0.shape[-1]
        self._dtype = X_0.dtype

        sobol_engine = SobolEngine(self._num_dim, scramble=True)
        self.X_samples = sobol_engine.draw(num_X_samples, dtype=self._dtype)

    def _variance(self, model):
        mvn = model.posterior(self.X_samples, observation_noise=True)
        if self.joint_variance:
            Y = self.get_posterior_samples(mvn).squeeze(dim=-1)
            vr = Y.var(dim=0)
        else:
            vr = mvn.variance.squeeze()
        return vr

    def _integrated_variance(self, model):
        return self._variance(model).mean(dim=-1)

    def _max(self, model, X):
        mvn = model.posterior(X, observation_noise=True)
        Y = self.get_posterior_samples(mvn).squeeze(dim=-1)
        Y_max = Y.max(dim=0)
        Y_maxmax = Y_max.values.max(dim=1)
        i = Y_max.indices[:, Y_maxmax.indices].diag()
        return torch.diagonal(Y[i, :, :], dim1=0, dim2=1).T

    def _mean_Y(self, model, X):
        mvn = model.posterior(X, observation_noise=True)
        return mvn.mean

    def _sd_max(self, model):
        mvn = model.posterior(self.X_samples, observation_noise=True)
        Y = self.get_posterior_samples(mvn).squeeze(dim=-1)
        # in lieu of argmax (which has no gradient)
        w = torch.exp(10 * Y).mean(dim=0)  # avg over Y
        w = w / w.sum(dim=-1, keepdims=True)  # norm over X

        X = self.X_samples.unsqueeze(0)
        w = w.unsqueeze(-1)
        mu = (w * X).sum(dim=1, keepdims=True)
        dx = X - mu
        vr = (w * dx**2).sum(dim=1)
        vr = vr.mean(dim=-1)  # mean over num_dim
        return torch.sqrt(vr)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        if True:
            Y_obs = self._max(self.model, X)
        else:
            Y_obs = self._mean_Y(self.model, X).squeeze(-1)

        q = X.shape[-2]
        if q == 1:
            Y_obs = Y_obs.unsqueeze(-1)
        model_f = self.model.condition_on_observations(X=X, Y=Y_obs)

        # sdm = self._sd_max(model_f)
        sd = torch.sqrt(self._integrated_variance(model_f))
        # Y_max_f = self._max(model_f, self.X_samples).max(dim=-1).values
        return Y_obs.max(dim=-1).values - sd
