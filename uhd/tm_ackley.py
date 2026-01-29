import math

import torch
from torch import Tensor, nn


class TMAckley(nn.Module):
    name = "tm_ackley"

    def __init__(self, num_dim: int, num_active: int, seed: int) -> None:
        super().__init__()
        assert isinstance(num_dim, int) and num_dim > 0
        assert isinstance(num_active, int) and 0 < num_active <= num_dim, (
            num_dim,
            num_active,
        )
        assert isinstance(seed, int)
        self.lb = -32.768
        self.ub = 32.768
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        idx = torch.randperm(num_dim, generator=g)[:num_active]
        self.parameters_ = nn.Parameter(
            self.lb + (self.ub - self.lb) * torch.rand(num_dim, generator=g)
        )
        self.register_buffer("active_idx", idx)
        self.register_buffer(
            "x_0_active",
            self.lb + (self.ub - self.lb) * torch.rand(num_active, generator=g),
        )

    def forward(self) -> Tensor:
        p = self.parameters_.index_select(0, self.active_idx)
        x_0 = self.x_0_active
        denom_neg = (x_0 - self.lb).clamp_min(1e-12)
        denom_pos = (self.ub - x_0).clamp_min(1e-12)
        y = torch.where(p < x_0, (p - x_0) / denom_neg, (p - x_0) / denom_pos)
        x = 32.768 * y
        d = x.numel()
        term1 = -20.0 * torch.exp(-0.2 * torch.sqrt(torch.sum(x * x) / d))
        term2 = -torch.exp(torch.sum(torch.cos(2.0 * math.pi * x)) / d)
        f = term1 + term2 + 20.0 + math.e
        return -f

    def get_param_accessor(self):
        from uhd.param_accessor import make_param_accessor

        return make_param_accessor(self)
