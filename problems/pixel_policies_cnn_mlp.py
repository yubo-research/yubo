"""CNN + MLP policy for dm_control-style pixel observations."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from policies.policy_mixin import PolicyParamsMixin
from problems.pixel_policies_encoders import (
    init_linear_and_conv,
    nature_cnn_encoder,
    obs_space_from_env_conf,
)


def _to_float_pixels(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(x):
        return x.to(dtype=torch.float32).div(255.0)
    x = x.to(dtype=torch.float32)
    if x.numel() > 0 and float(x.detach().amax().cpu()) > 1.0:
        return x.div(255.0)
    return x


class CNNMLPPolicyFactory:
    """Factory for CNN+MLP policy (pixel observations)."""

    def __init__(self, hidden_sizes, *, cnn_latent_dim: int = 256):
        self._hidden_sizes = tuple(hidden_sizes)
        self._cnn_latent_dim = int(cnn_latent_dim)

    def __call__(self, env_conf):
        return CNNMLPPolicy(env_conf, self._hidden_sizes, cnn_latent_dim=self._cnn_latent_dim)


class CNNMLPPolicy(PolicyParamsMixin, nn.Module):
    """Policy with CNN encoder for (H,W,C) pixel observations."""

    def __init__(self, env_conf, hidden_sizes, *, cnn_latent_dim: int = 256):
        super().__init__()
        self.problem_seed = env_conf.problem_seed
        self._env_conf = env_conf
        self._const_scale = 0.5

        obs_space = obs_space_from_env_conf(env_conf)
        assert hasattr(obs_space, "shape"), "Pixel policy expects Box observation space"
        shape = obs_space.shape
        assert len(shape) == 3, f"Expected (H,W,C), got {shape}"
        h, w, c = int(shape[0]), int(shape[1]), int(shape[2])
        assert h == 84 and w == 84, f"Expected 84x84 pixels, got {h}x{w}"
        num_action = int(env_conf.action_space.shape[0])

        self.encoder, feat_dim = nature_cnn_encoder(in_channels=c, latent_dim=cnn_latent_dim)
        dims = [feat_dim] + list(hidden_sizes) + [num_action]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.Tanh())
        self.head = nn.Sequential(*layers)

        self._init_params()
        with torch.inference_mode():
            self._flat_params_init = np.concatenate([p.data.detach().cpu().numpy().reshape(-1) for p in self.parameters()])

    def _init_params(self):
        init_linear_and_conv(self, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_float_pixels(x)
        if x.shape[-1] == 3:
            x = x.permute(*range(x.dim() - 3), -1, -3, -2)
        feats = self.encoder(x)
        return self.head(feats)

    def __call__(self, state):
        # state may be uint8 [0,255] or float [0,1].
        state = torch.as_tensor(state, dtype=torch.float32)
        if state.dim() == 3:
            state = state.unsqueeze(0)
        with torch.inference_mode():
            action = self.forward(state)
        return action.squeeze(0).detach().cpu().numpy()
