from dataclasses import dataclass
from typing import Any, Tuple

import torch
import torch.nn as nn

from uhd.uhd_collector import UHDCollector


@dataclass
class AdamWConfig:
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    metric: Any


def optimize_adamw(
    controller: nn.Module, collector: UHDCollector, num_rounds: int, config: AdamWConfig
) -> float:
    assert isinstance(controller, nn.Module)
    assert isinstance(collector, UHDCollector)
    assert isinstance(num_rounds, int) and num_rounds >= 0
    assert isinstance(config, AdamWConfig)
    assert isinstance(config.lr, float) and config.lr > 0.0
    assert isinstance(config.betas, tuple) and len(config.betas) == 2
    assert all(isinstance(b, float) for b in config.betas)
    assert isinstance(config.eps, float) and config.eps > 0.0
    assert isinstance(config.weight_decay, float) and config.weight_decay >= 0.0
    assert hasattr(config.metric, "measure")

    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    last_loss: float = float("nan")
    if num_rounds == 0:
        collector("DONE")
        return last_loss

    collector.start_eval()
    with torch.no_grad():
        initial_y_t = config.metric.measure(controller).detach()
    collector.stop_eval()
    initial_y = float(initial_y_t.cpu())
    if collector._y_best is None or initial_y > collector._y_best:
        collector._y_best = initial_y
    collector._dt_eval = None

    for _ in range(num_rounds):
        optimizer.zero_grad(set_to_none=True)
        collector.start_prop()
        loss_tensor = -config.metric.measure(controller)
        assert isinstance(loss_tensor, torch.Tensor) and loss_tensor.ndim == 0
        loss_tensor.backward()
        optimizer.step()
        collector.stop_prop()
        last_loss = float(loss_tensor.detach().cpu())

        collector.start_eval()
        with torch.no_grad():
            y_new_t = config.metric.measure(controller).detach()
        collector.stop_eval()
        y_new = float(y_new_t.cpu())
        collector.params(controller)
        collector.trace(y_new)

    collector("DONE")
    return last_loss
