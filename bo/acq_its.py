import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.utils import t_batch_mode_transform

# from IPython.core.debugger import set_trace
from torch.quasirandom import SobolEngine


class AcqITS(MCAcquisitionFunction):
    def __init__(self, model, num_X_samples=256, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self._num_X_samples = num_X_samples
        X_0 = self.model.train_inputs[0].detach()
        self._num_obs = X_0.shape[0]
        self._num_dim = X_0.shape[-1]
        self._dtype = X_0.dtype

        sobol_engine = SobolEngine(self._num_dim, scramble=True)
        if self._num_obs == 0:
            self.X_samples = sobol_engine.draw(num_X_samples, dtype=self._dtype)
        else:
            self.X_samples = self._sample_maxes(sobol_engine, num_X_samples)

    def _sample_maxes(self, sobol_engine, num_X_samples):
        X_obs = self.model.train_inputs[0]
        Y_obs = self.model.posterior(X_obs).mean.squeeze(-1)
        Y_max = Y_obs.max()
        # X_max = X_obs[Y_obs == Y_max]

        eps = 0.01
        X_samples = sobol_engine.draw(3 * num_X_samples, dtype=self._dtype)
        for _ in range(3):
            X = torch.maximum(torch.tensor(0.0), torch.minimum(torch.tensor(1.0), X_samples + eps * torch.randn(size=X_samples.shape)))
            Y = self.model.posterior(X, observation_noise=True).sample(torch.Size([num_X_samples])).squeeze(-1)
            Y, i = torch.max(Y, dim=1)
            i = i[Y > Y_max]
            X_samples = X[i]
        i = torch.randint(len(X_samples), (num_X_samples,))
        return X_samples[i]

    @t_batch_mode_transform()
    def forward(self, X):
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

        model_f = self.model.condition_on_observations(X=X, Y=self.model.posterior(X).mean)
        model_f.covar_module.base_kernel.lengthscale *= ((1 + num_obs) / (1 + num_obs + q)) ** (1.0 / num_dim)

        var_f = model_f.posterior(self.X_samples, observation_noise=True).variance.squeeze()
        if True:
            m = var_f.mean(dim=-1)
            s = var_f.std(dim=-1)
            return -(m + s)
        else:
            return -torch.exp(20 * var_f).mean(dim=-1)
