import torch
from botorch.models.model import Model

# from IPython.core.debugger import set_trace
from bo.acq_integrated import AcqIntegrated


class AcqIEI(AcqIntegrated):
    def __init__(self, model: Model, bounds: torch.Tensor, Y_max: torch.Tensor, **kwargs) -> None:
        super().__init__(model=model, bounds=bounds, **kwargs)
        self.register_buffer("Y_max", Y_max)

    def integrand(self, Y_samples: torch.Tensor, _) -> torch.Tensor:
        return torch.maximum(torch.tensor(0.0), Y_samples - self.Y_max)
