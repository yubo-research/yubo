from torch import nn


class MLPPolicyModule(nn.Module):
    def __init__(self, num_state: int, num_action: int, hidden_sizes=(32, 16)):
        super().__init__()
        dims = [num_state] + list(hidden_sizes) + [num_action]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
