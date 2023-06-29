import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform

# from IPython.core.debugger import set_trace
from torch import Tensor
from torch.quasirandom import SobolEngine


class AcqVM(MCAcquisitionFunction):
    """Soft entropy
    - uses softmax (numerator) as approximate
    - can't normalize b/c using a different arm (or set of arms) in
    each iteration of the optimizer
    - *can* compare unnormalized values b/c all based on mean & var, which
    have a consistent scale across iterations
    """

    def __init__(self, model: Model, beta, num_X_samples, num_Y_samples, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.beta = beta
        self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_Y_samples]))

        X_0 = self.model.train_inputs[0].detach()
        num_dim = X_0.shape[-1]
        dtype = X_0.dtype

        sobol_engine = SobolEngine(num_dim, scramble=True)
        self.X_samples = torch.cat((sobol_engine.draw(num_X_samples, dtype=dtype), self.model.train_inputs[0]), axis=0)
        mvn = self.model(self.X_samples)
        # self.weights = self._calc_p_max(self.model, self.X_samples, 1024)
        p = mvn.mean + mvn.variance / 2
        self.weights = torch.exp(beta * p)
        self.weights = self.weights / self.weights.sum()

    def _calc_p_max(self, model, X, num_px):
        posterior = model.posterior(X, posterior_transform=self.posterior_transform, observation_noise=True)
        Y = posterior.sample(torch.Size([num_px])).squeeze(dim=-1)  # num_Y_samples x b x len(X)
        return self._calc_p_max_from_Y(Y)

    def _calc_p_max_from_Y(self, Y):
        is_best = torch.argmax(Y, dim=-1)
        idcs, counts = torch.unique(is_best, return_counts=True)
        p_max = torch.zeros(Y.shape[-1])
        p_max[idcs] = counts / Y.shape[0]
        return p_max

    def _soft_entropy(self, Y):
        p_max = torch.exp(self.beta * Y).mean(dim=0)  # (mu + vr / 2))
        H = -(p_max * torch.log(p_max))
        while len(H.shape) > 1:
            H = H.mean(dim=-1)
        return H

    @t_batch_mode_transform()
    def _xxx_forward(self, X: Tensor) -> Tensor:
        self.to(device=X.device)

        model_f = self.model.condition_on_observations(X=X, Y=self.model.posterior(X).mean)
        mvn = model_f(self.X_samples)
        return -(self.weights * mvn.variance).sum(dim=-1)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        # mvn_0 = self.model(X)

        model_f = self.model.condition_on_observations(X=X, Y=self.model.posterior(X).mean)

        X_samples = torch.cat(
            (
                X,
                torch.tile(self.X_samples.unsqueeze(0), (X.shape[0], 1, 1)),
            ),
            axis=1,
        )
        mvn_f = model_f.posterior(X_samples, observation_noise=True)
        Y_f = self.get_posterior_samples(mvn_f).squeeze(dim=-1)

        p_max = torch.exp(self.beta * Y_f)
        p_max = p_max / p_max.sum(-1).unsqueeze(-1)
        p_max = p_max.mean(dim=0)

        q = X.shape[-2]
        p_max_0_f = p_max[:, :q]
        p_max_f = p_max[:, q:]

        H_0_f = -(p_max_0_f * torch.log(p_max_0_f)).mean(dim=-1)
        H_f = -(p_max_f * torch.log(p_max_f)).mean(dim=-1)

        # H_f  nice exploration, like IOPT
        # -H_0  optimizes ok
        # -H_0_f  optimizes better
        # H_0 = self._soft_entropy(mvn_0)
        # H_0_f = self._soft_entropy(mvn_0_f.mean, mvn_0_f.var)
        # H_f = self._soft_entropy(mvn_f.mean, mvn_f.var)

        return H_0_f - H_f
