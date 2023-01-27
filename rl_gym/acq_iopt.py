import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform

# from IPython.core.debugger import set_trace
from torch import Tensor
from torch.quasirandom import SobolEngine


class AcqIOpt(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        q: int = 1,
        num_samples: int = 256,
        seed: int = None,
    ) -> None:
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.q = q

        X_0 = self.model.train_inputs[0]
        num_dim = X_0.shape[-1]
        sobol_engine = SobolEngine(q * num_dim, scramble=True, seed=seed)
        X_samples = sobol_engine.draw(num_samples, dtype=X_0.dtype)
        X_samples = X_samples.view(num_samples, q, num_dim).to(device=X_0.device)
        self.register_buffer("X_samples", X_samples)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        assert X.shape[-2] == self.q, (X.shape[-1], self.q, "q must be set in the constructor")

        self.to(device=X.device)

        posterior = self.model.posterior(X)  # predictive posterior

        num_batches = X.shape[0]
        af = []
        for i_batch in range(num_batches):
            Y = posterior.mean[i_batch, :]  # q
            model_next = self.model.condition_on_observations(X=X[i_batch, ::], Y=Y)  # q x d

            samples_y_next = model_next.posterior(self.X_samples)
            integrated_variance_next = samples_y_next.variance.squeeze().mean()  # q

            af.append(-integrated_variance_next)

        return torch.stack(af)
