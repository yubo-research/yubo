import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform

# from IPython.core.debugger import set_trace
from torch import Tensor
from torch.quasirandom import SobolEngine


class AcqITS(MCAcquisitionFunction):
    def __init__(self, model: Model, num_X_samples=128, num_Y_samples=64, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self._num_X_samples = num_X_samples
        if num_Y_samples is not None:
            self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_Y_samples]))
        else:
            self.sampler = None

        X_0 = self.model.train_inputs[0].detach()
        self._num_obs = X_0.shape[0]
        self._num_dim = X_0.shape[-1]
        self._dtype = X_0.dtype

        sobol_engine = SobolEngine(self._num_dim, scramble=True)
        if self._num_obs == 0:
            self.X_samples = sobol_engine.draw(num_X_samples, dtype=self._dtype)
        else:
            self.X_samples = self._sample_maxes(num_X_samples)

    def _sample_maxes(self, num_X_samples):
        X_obs = self.model.train_inputs[0]
        Y_obs = self.model.posterior(X_obs).mean.squeeze(-1)
        Y_max = Y_obs.max()
        X_max = X_obs[Y_obs == Y_max]

        sobol_engine = SobolEngine(self._num_dim, scramble=True)

        X_samples = []
        while len(X_samples) < num_X_samples:
            X = sobol_engine.draw(num_X_samples, dtype=self._dtype)

            X = torch.cat(
                (
                    X_max,
                    X,
                )
            )
            Y = self.model.posterior(X, observation_noise=True).sample(torch.Size([num_X_samples])).squeeze(-1)
            y_m, i = torch.max(Y, dim=1)
            i = i[y_m > Y[:, 0]]
            X_samples.extend([X[ii] for ii in i])
        return torch.stack(X_samples[:num_X_samples])

    def _variance(self, model):
        mvn = model.posterior(self.X_samples, observation_noise=True)
        if self.sampler is not None:
            Y = self.get_posterior_samples(mvn).squeeze(dim=-1)
            vr = Y.var(dim=0)
        else:
            vr = mvn.variance.squeeze()
        return vr

    def _integrated_variance(self, model):
        return self._variance(model).mean(dim=-1)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        model_f = self.model.condition_on_observations(X=X, Y=self.model.posterior(X).mean)

        return -self._integrated_variance(model_f)
