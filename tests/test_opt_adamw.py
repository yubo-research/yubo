import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common.collector import Collector
from uhd.opt_adamw import AdamWConfig, optimize_adamw
from uhd.tm_mnist import TMMNIST
from uhd.uhd_collector import UHDCollector


class CETrainingMetric:
    def __init__(self, num_samples: int, batch_size: int, seed: int) -> None:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        x = torch.randn((num_samples, 1, 28, 28), generator=g)
        y = torch.randint(low=0, high=10, size=(num_samples,), generator=g)
        self.dataset = TensorDataset(x, y)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, generator=g)
        self.it = iter(self.loader)
        self.criterion = nn.CrossEntropyLoss()

    def measure(self, controller: nn.Module) -> torch.Tensor:
        try:
            xb, yb = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            xb, yb = next(self.it)
        device = next(controller.parameters()).device
        xb = xb.to(device)
        yb = yb.to(device)
        logits = controller(xb)
        return self.criterion(logits, yb)


def test_opt_adamw_trains_and_reduces_loss_cpu():
    torch.manual_seed(0)
    controller = TMMNIST(seed=123)
    metric = CETrainingMetric(num_samples=256, batch_size=64, seed=0)
    base_collector = Collector()
    collector = UHDCollector(name="tm_mnist", opt_name="adamw", collector=base_collector)
    config = AdamWConfig(
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        metric=metric,
    )
    with torch.no_grad():
        baseline_loss = float(metric.measure(controller))
    final_loss = optimize_adamw(controller, collector, num_rounds=10, config=config)
    assert final_loss < baseline_loss


def test_opt_adamw_device_migration_and_forward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    controller = TMMNIST(seed=7).to(device)
    metric = CETrainingMetric(num_samples=128, batch_size=32, seed=1)
    base_collector = Collector()
    collector = UHDCollector(name="tm_mnist", opt_name="adamw", collector=base_collector)
    config = AdamWConfig(
        lr=5e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        metric=metric,
    )
    final_loss = optimize_adamw(controller, collector, num_rounds=5, config=config)
    assert isinstance(final_loss, float)
    for p in controller.parameters():
        assert p.device.type == device
