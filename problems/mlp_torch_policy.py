from torch import nn


class MLPPolicyModule(nn.Module):
    def __init__(
        self,
        num_state: int,
        num_action: int,
        hidden_sizes=(32, 16),
        *,
        use_layer_norm: bool = False,
        tanh_out: bool = False,
    ):
        super().__init__()
        self.in_norm = nn.LayerNorm(num_state, elementwise_affine=True) if bool(use_layer_norm) else None
        dims = [num_state] + list(hidden_sizes) + [num_action]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if bool(tanh_out):
            layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.in_norm is not None:
            x = self.in_norm(x)
        return self.model(x)
