"""Shared CNN encoders and helpers for pixel policies."""

from __future__ import annotations

import torch
import torch.nn as nn


def obs_space_from_env_conf(env_conf):
    obs_space = getattr(env_conf, "state_space", None)
    if obs_space is None:
        raise ValueError("Observation space is missing on env_conf. Call env_conf.ensure_spaces() before building pixel policies.")
    return obs_space


def init_linear_and_conv(module: nn.Module, *, gain: float) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def tiny_atari_cnn_encoder(in_channels: int = 4) -> tuple[nn.Module, int]:
    """Tiny CNN for Atari: ~10k params. 4->4->8->8 convs, small head."""
    encoder = nn.Sequential(
        nn.Conv2d(in_channels, 4, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(4, 8, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(8, 8, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(7),
        nn.Flatten(),
    )
    with torch.inference_mode():
        out_dim = encoder(torch.zeros(1, in_channels, 84, 84)).shape[-1]
    return encoder, out_dim


def nature_cnn_encoder(in_channels: int = 3, latent_dim: int = 256) -> tuple[nn.Module, int]:
    """Nature DQN-style CNN: 3 conv layers + flatten -> latent_dim."""
    layers = [
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
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
        dummy = torch.zeros(1, in_channels, 84, 84)
        out_dim = encoder(dummy).shape[-1]
    if latent_dim != out_dim:
        encoder = nn.Sequential(encoder, nn.Linear(out_dim, latent_dim), nn.ReLU())
        out_dim = latent_dim
    return encoder, out_dim
