import torch
from torch import Tensor, nn


class TMSphere(nn.Module):
    name = "tm_sphere"

    def __init__(self, num_dim: int, num_active: int, seed: int) -> None:
        super().__init__()
        assert isinstance(num_dim, int) and num_dim > 0
        assert isinstance(num_active, int) and 0 < num_active <= num_dim, (
            num_dim,
            num_active,
        )
        assert isinstance(seed, int)
        self.lb = 0.0
        self.ub = 1.0
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        idx = torch.randperm(num_dim, generator=g)[:num_active]
        self.register_buffer("x_0", torch.rand((), generator=g))
        self.parameters_ = nn.Parameter(torch.rand(num_dim, generator=g))
        self.register_buffer("active_idx", idx)

    def forward(self) -> Tensor:
        x = self.parameters_.index_select(0, self.active_idx) - self.x_0
        return -torch.sum(x * x)

    def get_param_accessor(self):
        from uhd.param_accessor import make_param_accessor

        return make_param_accessor(self)
