import torch
from torch import Tensor, nn


class MNISTClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.classifier(x)


class TMMNIST(nn.Module):
    name = "tm_mnist"

    def __init__(self, seed: int) -> None:
        super().__init__()
        assert isinstance(seed, int)
        torch.manual_seed(seed)
        self.lb = -2.0
        self.ub = 2.0
        self.net = MNISTClassifier()

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def get_param_accessor(self):
        from uhd.param_accessor import make_param_accessor

        return make_param_accessor(self)
