from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform

# from IPython.core.debugger import set_trace
from torch import Tensor


class AcqVar(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
    ) -> None:
        super(AnalyticAcquisitionFunction, self).__init__(model=model)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        return self.model.posterior(X).variance.squeeze()
