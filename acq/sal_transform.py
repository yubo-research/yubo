from typing import Optional

import torch
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch import Module as GPyTorchModule
from gpytorch.constraints import GreaterThan
from gpytorch.priors import Prior
from torch import Tensor, nn


# See G. Rios, F. Tobar, Compositionally-Warped Gaussian Processes, https://arxiv.org/pdf/1906.09665
# See also ChatGPT, who wrote this code.
# Code is modeled on botorch.models.transforms.input.Warp
class SALTransform(OutcomeTransform, GPyTorchModule):
    def __init__(
        self,
        batch_shape: Optional[torch.Size] = None,
        eps_positive=1e-6,
        a_prior: Optional[Prior] = None,
        b_prior: Optional[Prior] = None,
        c_prior: Optional[Prior] = None,
        d_prior: Optional[Prior] = None,
    ):
        super().__init__()

        self.batch_shape = batch_shape or torch.Size([])
        if len(self.batch_shape) > 0:
            batch_shape = self.batch_shape + torch.Size([1])
        else:
            batch_shape = self.batch_shape

        for p_name in ["a", "b", "c", "d"]:
            self.register_parameter(
                p_name,
                nn.Parameter(torch.full(batch_shape, 1.0)),
            )

        for prior, name in [
            (a_prior, "a_prior"),
            (b_prior, "b_prior"),
            (c_prior, "c_prior"),
            (d_prior, "d_prior"),
        ]:
            if prior is not None:
                self.register_prior(
                    name,
                    prior,
                    lambda m: getattr(m, name.split("_")[0]),
                    lambda m, v: m._set(name, v),
                )

        for p_name, initial_value in [
            ("b", 1.0),
            ("c", 1.0),
        ]:
            constraint = GreaterThan(
                eps_positive,
                transform=None,
                # set the initial value to be the identity transformation
                initial_value=torch.tensor(initial_value),
            )
            self.register_constraint(param_name=p_name, constraint=constraint)

    def _set(self, name, value):
        self.initialize(**{name: value})

    def forward(self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        sinh_arg = self.c * torch.arcsinh(Y) - self.d
        cosh_term = torch.cosh(sinh_arg) ** 2
        denom = 1 + Y**2
        Y_new = self.a + self.b * torch.sinh(sinh_arg)
        Yvar_new = (self.b**2) * (self.c**2) * cosh_term * (Yvar / denom)
        return Y_new, Yvar_new

    def untransform(self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        Y_new = torch.sinh((torch.arcsinh((Y - self.a) / self.b) + self.d) / self.c)

        sinh_arg = self.c * torch.arcsinh(Y_new) - self.d
        cosh_term = torch.cosh(sinh_arg) ** 2
        denom = 1 + Y_new**2
        Yvar_new = Yvar * denom / ((self.b**2) * (self.c**2) * cosh_term)

        return Y_new, Yvar_new
