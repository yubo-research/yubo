import torch

from uhd.mnist_metric import MNISTMetric
from uhd.tm_mnist import TMMNIST


def test_mnist_metric_returns_negative_ce_and_tensor(tmp_path):
    controller = TMMNIST(seed=0)
    metric = MNISTMetric(
        data_root=str(tmp_path), batch_size=16, seed=0, train=True, num_workers=0
    )
    val = metric.measure(controller)
    assert isinstance(val, torch.Tensor)
    assert val.ndim == 0
    assert float(val) <= 0.0


def test_mnist_metric_device_migration(tmp_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    controller = TMMNIST(seed=1).to(device)
    metric = MNISTMetric(
        data_root=str(tmp_path), batch_size=8, seed=1, train=True, num_workers=0
    )
    out = metric.measure(controller)
    assert isinstance(out, torch.Tensor)
    for p in controller.parameters():
        assert p.device.type == device
