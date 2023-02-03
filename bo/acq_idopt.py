import torch
from botorch.models.model import Model

# from IPython.core.debugger import set_trace
from bo.acq_integrated import AcqIntegrated


class AcqIDOpt(AcqIntegrated):
    def __init__(self, model: Model, bounds: torch.Tensor, X_max: torch.Tensor = None, **kwargs) -> None:
        super().__init__(model=model, bounds=bounds, X_special=X_max, **kwargs)

    def integrand(self, Y_samples: torch.Tensor, Y_special: torch.Tensor) -> torch.Tensor:
        if Y_special is not None:
            delta = Y_special - Y_samples
        else:
            delta = Y_samples

        return delta.var(axis=0)  # num_X_samples; var over Y
