"""Atari pixel policies with a Gaussian MLP head (split from cnn module for concrete_types)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from policies.policy_mixin import PolicyParamsMixin
from problems.pixel_atari_spatial import (
    atari_in_channels_from_obs_shape,
    atari_obs_to_nchw,
)
from problems.pixel_policies_encoders import (
    init_linear_and_conv,
    nature_cnn_encoder,
    obs_space_from_env_conf,
    tiny_atari_cnn_encoder,
)


class AtariGaussianPolicyFactory:
    def __init__(
        self,
        hidden_sizes=(16, 16),
        *,
        cnn_latent_dim: int = 64,
        variant: str = "small",
        deterministic_eval: bool = True,
        init_log_std: float = -0.5,
    ):
        self._hidden_sizes = tuple(hidden_sizes)
        self._cnn_latent_dim = int(cnn_latent_dim)
        self._variant = str(variant).lower()
        self._deterministic_eval = bool(deterministic_eval)
        self._init_log_std = float(init_log_std)

    def __call__(self, env_conf):
        return AtariGaussianPolicy(
            env_conf,
            self._hidden_sizes,
            cnn_latent_dim=self._cnn_latent_dim,
            variant=self._variant,
            deterministic_eval=self._deterministic_eval,
            init_log_std=self._init_log_std,
        )


class AtariGaussianPolicy(PolicyParamsMixin, nn.Module):
    def __init__(
        self,
        env_conf,
        hidden_sizes,
        *,
        cnn_latent_dim: int = 64,
        variant: str = "small",
        deterministic_eval: bool = True,
        init_log_std: float = -0.5,
    ):
        super().__init__()
        self.problem_seed = env_conf.problem_seed
        self._env_conf = env_conf
        self._const_scale = 0.5
        self._deterministic_eval = bool(deterministic_eval)

        obs_space = obs_space_from_env_conf(env_conf)
        shape = obs_space.shape
        in_channels = atari_in_channels_from_obs_shape(shape, require_spatial_84=False)
        num_actions = int(env_conf.action_space.n)

        if variant == "small":
            self.encoder, feat_dim = tiny_atari_cnn_encoder(in_channels=in_channels)
        else:
            self.encoder, feat_dim = nature_cnn_encoder(in_channels=in_channels, latent_dim=cnn_latent_dim)
        dims = [feat_dim] + list(hidden_sizes) + [num_actions]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.head = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.full((num_actions,), float(init_log_std), dtype=torch.float32))
        self._min_log_std = -20.0
        self._max_log_std = 2.0

        self._init_params()
        with torch.inference_mode():
            self._flat_params_init = np.concatenate([p.data.detach().cpu().numpy().reshape(-1) for p in self.parameters()])

    def _init_params(self):
        init_linear_and_conv(self, gain=0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = atari_obs_to_nchw(x)
        feats = self.encoder(x)
        mean = self.head(feats)
        log_std = torch.clamp(self.log_std, self._min_log_std, self._max_log_std)
        return mean, log_std

    def __call__(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        with torch.inference_mode():
            mean, log_std = self.forward(state)
            if self._deterministic_eval:
                action = mean.argmax(dim=-1).squeeze(0)
            else:
                std = torch.exp(log_std)
                sample = mean + std * torch.randn_like(mean)
                action = sample.argmax(dim=-1).squeeze(0)
        return int(action.item())
