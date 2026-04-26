"""Atari CNN and Gaussian-head pixel policies."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from problems.pixel_atari_spatial import atari_in_channels_from_obs_shape, atari_obs_to_nchw
from problems.pixel_policies_encoders import init_linear_and_conv, nature_cnn_encoder, obs_space_from_env_conf, tiny_atari_cnn_encoder
from problems.policy_mixin import PolicyParamsMixin


class AtariCNNPolicyFactory:
    """Factory for CNN policy with discrete actions (Atari).
    variant='small' -> ~10k params; default -> ~377k params."""

    def __init__(
        self,
        hidden_sizes=(512,),
        *,
        cnn_latent_dim: int = 512,
        variant: str = "default",
    ):
        self._hidden_sizes = tuple(hidden_sizes)
        self._cnn_latent_dim = int(cnn_latent_dim)
        self._variant = str(variant).lower()

    def __call__(self, env_conf):
        return AtariCNNPolicy(
            env_conf,
            self._hidden_sizes,
            cnn_latent_dim=self._cnn_latent_dim,
            variant=self._variant,
        )


class AtariCNNPolicy(PolicyParamsMixin, nn.Module):
    """CNN policy for Atari: (4,84,84) or (84,84,4) stacked frames -> discrete action."""

    def __init__(
        self,
        env_conf,
        hidden_sizes,
        *,
        cnn_latent_dim: int = 512,
        variant: str = "default",
    ):
        super().__init__()
        self.problem_seed = env_conf.problem_seed
        self._env_conf = env_conf
        self._const_scale = 0.5

        obs_space = obs_space_from_env_conf(env_conf)
        shape = obs_space.shape
        in_channels = atari_in_channels_from_obs_shape(shape, require_spatial_84=True)

        num_actions = int(env_conf.action_space.n)
        if str(variant).lower() == "small":
            self.encoder, feat_dim = tiny_atari_cnn_encoder(in_channels=in_channels)
            dims = [feat_dim] + list(hidden_sizes) + [num_actions] if hidden_sizes else [feat_dim, 24, num_actions]
        else:
            self.encoder, feat_dim = nature_cnn_encoder(in_channels=in_channels, latent_dim=cnn_latent_dim)
            dims = [feat_dim] + list(hidden_sizes) + [num_actions]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.head = nn.Sequential(*layers)

        self._init_params()
        with torch.inference_mode():
            self._flat_params_init = np.concatenate([p.data.detach().cpu().numpy().reshape(-1) for p in self.parameters()])

    def _init_params(self):
        init_linear_and_conv(self, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = atari_obs_to_nchw(x)
        feats = self.encoder(x)
        return self.head(feats)

    def __call__(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        if state.dim() == 3:
            state = state.unsqueeze(0)
        with torch.inference_mode():
            logits = self.forward(state)
        action = logits.argmax(dim=-1).squeeze(0)
        return int(action.item())
