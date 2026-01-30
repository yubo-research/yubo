import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform

# from IPython.core.debugger import set_trace
from torch import Tensor


class AcqMinDist(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        toroidal: bool,
        X_max: torch.Tensor = None,
    ) -> None:
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        if toroidal:
            self._delta_squared = self._toroidal_delta_squared
        self.X_max = X_max

    def _toroidal_delta_squared(self, x, y):
        d = torch.abs(x - y)
        flip = d > 0.5
        d[flip] = 1 - d[flip]
        return d**2

    def _delta_squared(self, x, y):
        return (x - y) ** 2

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        # posterior = self.model.posterior(X)
        num_batches = X.shape[0]
        af = []
        for i_batch in range(num_batches):
            # Y = posterior.mean[i_batch, :]  # q
            dist2 = self._delta_squared(X[i_batch, ::], self.model.train_inputs[0]).sum(
                axis=-1
            )
            md = dist2.min()
            # md = torch.sqrt(1e-9 + md)
            if self.X_max is not None:
                dist2_max = self._delta_squared(self.X_max, X[i_batch, ::]).sum(axis=-1)
                # md = md - torch.sqrt(dist2_max.squeeze())
                md = md - dist2_max.squeeze()
            af.append(md)
        return torch.stack(af)
