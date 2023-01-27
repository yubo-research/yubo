import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform

# from IPython.core.debugger import set_trace
from torch import Tensor


class AcqMinDist(AnalyticAcquisitionFunction):
    def __init__(self, model: Model, toroidal: bool) -> None:
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        if toroidal:
            self._distance_squared = self._toroidal_distance_squared

    def _toroidal_distance_squared(self, x, y):
        d = torch.abs(x - y)
        flip = d > 0.5
        d[flip] = 1 - d[flip]
        return d**2

    def _distance_squared(self, x, y):
        return (x - y) ** 2

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        num_batches = X.shape[0]
        af = []
        for i_batch in range(num_batches):
            dist2 = self._distance_squared(X[i_batch, ::], self.model.train_inputs[0]).sum(axis=-1)
            af.append(dist2.min())
        return torch.stack(af)
