from dataclasses import dataclass
from typing import Any, Callable

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


@dataclass(frozen=True)
class NetworkSpec:
    backbone: BackboneSpec
    head: HeadSpec


@dataclass(frozen=True)
class ActorCriticSpec:
    actor: NetworkSpec
    critic: NetworkSpec

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        actor_backbone_name: str,
        critic_backbone_name: str | None = None,
    ) -> "ActorCriticSpec":
        def pick(name, default):
            value = getattr(config, name, None)
            return default if value is None else value

        critic_name = str(critic_backbone_name if critic_backbone_name is not None else pick("critic_backbone_name", actor_backbone_name))
        return cls(
            actor=NetworkSpec(
                backbone=BackboneSpec(
                    name=str(actor_backbone_name),
                    hidden_sizes=tuple(config.backbone_hidden_sizes),
                    activation=str(config.backbone_activation),
                    layer_norm=bool(config.backbone_layer_norm),
                ),
                head=HeadSpec(
                    hidden_sizes=tuple(config.actor_head_hidden_sizes),
                    activation=str(pick("actor_head_activation", config.head_activation)),
                ),
            ),
            critic=NetworkSpec(
                backbone=BackboneSpec(
                    name=critic_name,
                    hidden_sizes=tuple(pick("critic_backbone_hidden_sizes", config.backbone_hidden_sizes)),
                    activation=str(pick("critic_backbone_activation", config.backbone_activation)),
                    layer_norm=bool(pick("critic_backbone_layer_norm", config.backbone_layer_norm)),
                ),
                head=HeadSpec(
                    hidden_sizes=tuple(config.critic_head_hidden_sizes),
                    activation=str(pick("critic_head_activation", config.head_activation)),
                ),
            ),
        )


_BACKBONES: dict[str, Callable[[BackboneSpec, int], tuple[nn.Module, int]]] = {}


def register_backbone(name: str):
    def _decorator(builder):
        if name in _BACKBONES:
            raise ValueError(f"Backbone '{name}' already registered.")
        _BACKBONES[name] = builder
        return builder

    return _decorator


def _activation(name: str) -> type[nn.Module]:
    if isinstance(name, type) and issubclass(name, nn.Module):
        return name
    key = "".join(ch for ch in str(name).lower() if ch.isalnum())
    act = next(
        (
            mod
            for raw, mod in vars(nn.modules.activation).items()
            if isinstance(mod, type) and issubclass(mod, nn.Module) and key == "".join(ch for ch in str(raw).lower() if ch.isalnum())
        ),
        None,
    )
    if act is None:
        raise ValueError(f"Unsupported activation '{name}'.")
    return act


@register_backbone("nature_cnn")
def _build_nature_cnn(spec: BackboneSpec, input_dim: int) -> tuple[nn.Module, int]:
    return _build_nature_cnn_with_channels(spec, input_dim, in_channels=3)


def _build_nature_cnn_with_channels(spec: BackboneSpec, input_dim: int, *, in_channels: int) -> tuple[nn.Module, int]:
    del input_dim
    act = _activation(spec.activation)
    layers = [
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        act(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        act(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        act(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    ]
    encoder = nn.Sequential(*layers)
    with torch.inference_mode():
        dummy = torch.zeros(1, in_channels, 84, 84)
        out_dim = int(encoder(dummy).shape[-1])
    return (encoder, out_dim)


def build_nature_cnn_encoder(
    *,
    in_channels: int,
    activation: str = "relu",
    latent_dim: int | None = None,
) -> tuple[nn.Module, int]:
    spec = BackboneSpec(name="nature_cnn", hidden_sizes=(), activation=str(activation), layer_norm=False)
    encoder, out_dim = _build_nature_cnn_with_channels(spec, 0, in_channels=int(in_channels))
    if latent_dim is None or int(latent_dim) == int(out_dim):
        return (encoder, int(out_dim))
    act = _activation(activation)
    projected = nn.Sequential(encoder, nn.Linear(int(out_dim), int(latent_dim)), act())
    return (projected, int(latent_dim))


@register_backbone("nature_cnn_atari")
def _build_nature_cnn_atari(spec: BackboneSpec, input_dim: int) -> tuple[nn.Module, int]:
    return _build_nature_cnn_with_channels(spec, input_dim, in_channels=4)


@register_backbone("mlp")
def _build_mlp(spec: BackboneSpec, input_dim: int) -> tuple[nn.Module, int]:
    layers: list[nn.Module] = []
    if spec.layer_norm:
        layers.append(nn.LayerNorm(input_dim, elementwise_affine=True))
    dims = [int(input_dim)] + [int(h) for h in spec.hidden_sizes]
    if len(dims) == 1:
        module = nn.Sequential(*layers) if layers else nn.Identity()
        return (module, int(input_dim))
    act = _activation(spec.activation)
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(act())
    return (nn.Sequential(*layers), int(dims[-1]))


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


def init_linear_layers(module: nn.Module, *, gain: float = 0.5) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=float(gain))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
