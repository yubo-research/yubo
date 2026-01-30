from typing import Optional

import torch
from gpytorch import Module as GPyTorchModule
from gpytorch.constraints import GreaterThan
from gpytorch.priors import Prior
from torch import Tensor, nn


# This is a simpler (two parameters instead of four) take on the SAL transform from
#   G. Rios, F. Tobar, Compositionally-Warped Gaussian Processes, https://arxiv.org/pdf/1906.09665
# Code follows (steals from) botorch.models.transforms.input.Warp
class YTransform(GPyTorchModule):
    def __init__(
        self,
        batch_shape: Optional[torch.Size] = None,
        eps_positive=1e-6,
        a_prior: Optional[Prior] = None,
        b_prior: Optional[Prior] = None,
    ):
        super().__init__()

        self.batch_shape = batch_shape or torch.Size([])
        self._eps_positive = eps_positive

        if len(self.batch_shape) > 0:
            batch_shape = self.batch_shape + torch.Size([1])
        else:
            batch_shape = self.batch_shape

        # Initialize at roughly an identity transformation
        a_init = 0.1
        b_init = 0.0

        self.register_parameter("a", nn.Parameter(torch.full(batch_shape, a_init)))
        self.register_parameter("b", nn.Parameter(torch.full(batch_shape, b_init)))

        for prior, name in [
            (a_prior, "a_prior"),
            (b_prior, "b_prior"),
        ]:
            if prior is not None:

                def the_prior(m, name=name):
                    return getattr(m, name.split("_")[0])

                def setter(m, v, name=name):
                    self.initialize(**{name: v})

                self.register_prior(
                    name,
                    prior,
                    the_prior,
                    setter,
                )

        self.register_constraint(
            param_name="a",
            constraint=GreaterThan(
                eps_positive,
                transform=None,
                initial_value=torch.tensor(a_init),
            ),
        )

    def forward(self, Y: Tensor) -> Tensor:
        if len(Y) <= 1:
            return Y
        y_range = Y.amax() - Y.amin()
        if y_range == 0:
            return Y
        Y = (Y - Y.amin()) / y_range
        Y = 2 * Y - 1
        Y = torch.tanh(self.a * Y) / self.a
        Y = torch.clip(Y, -1 + self._eps_positive, 1 - self._eps_positive)
        Y = torch.arctanh(Y) + self.b

        return Y
