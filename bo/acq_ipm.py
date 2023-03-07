import torch
from botorch.acquisition import ExpectedImprovement
from botorch.models.model import Model
from IPython.core.debugger import set_trace

from bo.acq_integrated import AcqIntegrated


class AcqIPM(AcqIntegrated):
    def __init__(self, model: Model, bounds: torch.Tensor, Y_max: torch.Tensor, **kwargs) -> None:
        super().__init__(model=model, bounds=bounds, **kwargs)
        self.ei = ExpectedImprovement(model, best_f=Y_max)
        self.register_buffer("Y_max", Y_max)

    def fantasy_observation(self, X: torch.Tensor) -> torch.Tensor:
        Y_obs = self.ei(X)
        set_trace()
        return Y_obs

    def integrand(self, Y_samples: torch.Tensor, _) -> torch.Tensor:  # num_Y_samples x num_X_samples
        return torch.maximum(torch.tensor(0.0), Y_samples - self.Y_max)
