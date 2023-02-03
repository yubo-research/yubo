import torch
from botorch.models.model import Model

# from IPython.core.debugger import set_trace
from bo.acq_integrated import AcqIntegrated


class AcqIUCB(AcqIntegrated):
    def __init__(self, model: Model, bounds: torch.Tensor, beta: float = 1.0, **kwargs) -> None:
        super().__init__(model=model, bounds=bounds, **kwargs)
        self.register_buffer("sq_beta", torch.sqrt(torch.as_tensor(float(beta))))

    def integrand(self, Y_samples: torch.Tensor, _) -> torch.Tensor:  # num_Y_samples x num_X_samples
        mu = Y_samples.mean(axis=0)
        std = Y_samples.std(axis=0)
        return mu + self.sq_beta * std
