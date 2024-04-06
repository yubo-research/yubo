import torch
from botorch.models import SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_normal_samples


class AppxNormal:
    def __init__(self, model: SingleTaskGP, num_X_samples, num_Y_samples=128):
        self._model = model

        X = model.train_inputs[0]
        assert len(X.shape) == 2, (X.shape, "I only handle single-output models")
        self.num_dim = X.shape[1]
        self.device = X.device

        self._X_base_samples = draw_sobol_normal_samples(
            self.num_dim,
            num_X_samples,
            device=self.device,
        )
        self._p_x = torch.exp(-(self._X_base_samples**2).sum(dim=1) / 2)
        self._p_x = self._p_x / self._p_x.sum()
        assert self._p_x.shape == (num_X_samples,), (self._p_x.shape, num_X_samples)

        self._sampler_y = SobolQMCNormalSampler(sample_shape=torch.Size([num_Y_samples]))

    def evaluate(self, mu, sigma):
        p_star = self._mk_p_star(self._mk_normal(mu, sigma))
        assert p_star.shape == self._p_x.shape, (p_star.shape, self._p_x.shape)
        return ((p_star - self._p_x) ** 2).sum()

    def _mk_normal(self, mu, sigma):
        return mu + sigma * self._X_base_samples

    def _mk_p_star(self, X):
        mvn = self._model.posterior(X, observation_noise=False)
        Y = self._sampler_y(mvn).squeeze(-1)

        z = torch.zeros(size=Y.shape)
        i = Y.max(dim=1).indices
        j = torch.arange(128)
        z[j, i] = 1
        p_star = z.mean(dim=0)
        p_star = p_star / p_star.sum()
        return p_star
