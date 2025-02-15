import torch
from botorch.models.transforms.outcome import OutcomeTransform
from torch import Tensor


class SALTransform(OutcomeTransform):
    def __init__(self, a: float, b: float, c: float, d: float):
        super().__init__()
        self.a = torch.tensor(a, dtype=torch.float32)
        self.b = torch.tensor(b, dtype=torch.float32)
        self.c = torch.tensor(c, dtype=torch.float32)
        self.d = torch.tensor(d, dtype=torch.float32)

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
