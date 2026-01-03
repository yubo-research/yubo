from dataclasses import dataclass
from typing import Any, Callable, Type


@dataclass
class TargetSpec:
    requires_dims: bool
    controller_type_factory: Callable[[], Type[Any]]
    make_adamw_metric: Callable[[], Any]
    make_bo_metric: Callable[[], Any]


def _make_sphere_type() -> Type[Any]:
    from uhd.tm_sphere import TMSphere

    return TMSphere


def _make_ackley_type() -> Type[Any]:
    from uhd.tm_ackley import TMAckley

    return TMAckley


def _make_mnist_type() -> Type[Any]:
    from uhd.tm_mnist import TMMNIST

    return TMMNIST


def _make_scalar_metric() -> Any:
    import torch.nn as nn  # noqa: F401

    class _AdamWMetric:
        def measure(self, controller: nn.Module):
            return controller()

    return _AdamWMetric()


def _make_mnist_metric() -> Any:
    from uhd.mnist_metric import MNISTMetric

    return MNISTMetric(data_root="./data", batch_size=64, seed=0, train=True, num_workers=0)


TARGET_SPECS = {
    "tm_sphere": TargetSpec(
        requires_dims=True,
        controller_type_factory=_make_sphere_type,
        make_adamw_metric=_make_scalar_metric,
        make_bo_metric=_make_scalar_metric,
    ),
    "tm_ackley": TargetSpec(
        requires_dims=True,
        controller_type_factory=_make_ackley_type,
        make_adamw_metric=_make_scalar_metric,
        make_bo_metric=_make_scalar_metric,
    ),
    "tm_mnist": TargetSpec(
        requires_dims=False,
        controller_type_factory=_make_mnist_type,
        make_adamw_metric=_make_mnist_metric,
        make_bo_metric=_make_mnist_metric,
    ),
}
