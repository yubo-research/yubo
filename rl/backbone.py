from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass
class BackboneSpec:
    name: str = "mlp"
    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "silu"
    layer_norm: bool = True


@dataclass
class HeadSpec:
    hidden_sizes: tuple[int, ...] = ()
    activation: str = "silu"


_BACKBONES: dict[str, Callable[[BackboneSpec, int], tuple[nn.Module, int]]] = {}


def register_backbone(name: str):
    def _decorator(builder):
        if name in _BACKBONES:
            raise ValueError(f"Backbone '{name}' already registered.")
        _BACKBONES[name] = builder
        return builder

    return _decorator


def _activation(name: str) -> type[nn.Module]:
    key = str(name).strip().lower()
    if key in ("silu", "swish"):
        return nn.SiLU
    if key in ("relu",):
        return nn.ReLU
    if key in ("tanh",):
        return nn.Tanh
    raise ValueError(f"Unsupported activation '{name}'.")


@register_backbone("nature_cnn")
def _build_nature_cnn(spec: BackboneSpec, input_dim: int) -> tuple[nn.Module, int]:
    """Nature DQN-style CNN for (C,H,W) pixel input. input_dim ignored; uses 3x84x84."""
    del input_dim  # pixel shape is fixed
    layers = [
        nn.Conv2d(3, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    ]
    encoder = nn.Sequential(*layers)
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, 84, 84)
        out_dim = int(encoder(dummy).shape[-1])
    return encoder, out_dim


@register_backbone("nature_cnn_atari")
def _build_nature_cnn_atari(spec: BackboneSpec, input_dim: int) -> tuple[nn.Module, int]:
    """Nature CNN for Atari: 4 stacked grayscale frames (C,H,W) = (4,84,84)."""
    del input_dim
    layers = [
        nn.Conv2d(4, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    ]
    encoder = nn.Sequential(*layers)
    with torch.inference_mode():
        dummy = torch.zeros(1, 4, 84, 84)
        out_dim = int(encoder(dummy).shape[-1])
    return encoder, out_dim


@register_backbone("mlp")
def _build_mlp(spec: BackboneSpec, input_dim: int) -> tuple[nn.Module, int]:
    layers: list[nn.Module] = []
    if spec.layer_norm:
        layers.append(nn.LayerNorm(input_dim, elementwise_affine=True))

    dims = [int(input_dim)] + [int(h) for h in spec.hidden_sizes]
    if len(dims) == 1:
        module = nn.Sequential(*layers) if layers else nn.Identity()
        return module, int(input_dim)

    act = _activation(spec.activation)
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(act())
    return nn.Sequential(*layers), int(dims[-1])


def build_backbone(spec: BackboneSpec, input_dim: int) -> tuple[nn.Module, int]:
    if spec.name not in _BACKBONES:
        raise ValueError(f"Unknown backbone '{spec.name}'. Available: {sorted(_BACKBONES)}")
    return _BACKBONES[spec.name](spec, int(input_dim))


def build_mlp_head(spec: HeadSpec, input_dim: int, output_dim: int) -> nn.Module:
    dims = [int(input_dim)] + [int(h) for h in spec.hidden_sizes] + [int(output_dim)]
    if len(dims) == 2:
        return nn.Linear(dims[0], dims[1])

    layers: list[nn.Module] = []
    act = _activation(spec.activation)
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(act())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)
