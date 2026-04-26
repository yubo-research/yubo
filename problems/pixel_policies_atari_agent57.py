"""Agent57-inspired recurrent Atari policy."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from problems.pixel_atari_spatial import atari_in_channels_from_obs_shape, atari_obs_to_nchw
from problems.pixel_policies_encoders import nature_cnn_encoder, obs_space_from_env_conf, tiny_atari_cnn_encoder
from problems.policy_mixin import PolicyParamsMixin


class AtariAgent57LiteFactory:
    """Factory for Agent57-inspired policy: CNN torso + LSTM + Q-head. Small (~15–25k params)."""

    def __init__(
        self,
        lstm_hidden: int = 32,
        *,
        cnn_variant: str = "small",
    ):
        self._lstm_hidden = int(lstm_hidden)
        self._cnn_variant = str(cnn_variant).lower()

    def __call__(self, env_conf):
        return AtariAgent57LitePolicy(
            env_conf,
            lstm_hidden=self._lstm_hidden,
            cnn_variant=self._cnn_variant,
        )


class AtariAgent57LitePolicy(PolicyParamsMixin, nn.Module):
    """Agent57-inspired: CNN torso -> LSTM(conv_feat, prev_action, prev_reward) -> Q-head -> argmax."""

    _recurrent = True  # needs prev_action, prev_reward in rollout

    def __init__(self, env_conf, lstm_hidden: int = 32, *, cnn_variant: str = "small"):
        super().__init__()
        self.problem_seed = env_conf.problem_seed
        self._env_conf = env_conf
        self._const_scale = 0.5

        obs_space = obs_space_from_env_conf(env_conf)
        shape = obs_space.shape
        in_channels = atari_in_channels_from_obs_shape(shape, require_spatial_84=False)
        num_actions = int(env_conf.action_space.n)

        if cnn_variant == "small":
            self.encoder, conv_dim = tiny_atari_cnn_encoder(in_channels=in_channels)
        else:
            self.encoder, conv_dim = nature_cnn_encoder(in_channels=in_channels, latent_dim=64)
        self.proj = nn.Linear(conv_dim, lstm_hidden)
        lstm_input_dim = lstm_hidden + num_actions + 1
        self.lstm = nn.LSTM(int(lstm_input_dim), lstm_hidden, batch_first=True)
        self.q_head = nn.Linear(lstm_hidden, num_actions)

        self._num_actions = num_actions
        self._lstm_hidden = lstm_hidden
        self._h = self._c = None

        self._init_params()
        with torch.inference_mode():
            self._flat_params_init = np.concatenate([p.data.detach().cpu().numpy().reshape(-1) for p in self.parameters()])

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p, gain=0.5)
            else:
                nn.init.zeros_(p)

    def reset_state(self):
        self._h = self._c = None

    def forward(
        self,
        x: torch.Tensor,
        prev_action: int = 0,
        prev_reward: float = 0.0,
        h=None,
        c=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = atari_obs_to_nchw(x)
        conv_feat = self.encoder(x)
        conv_feat = self.proj(conv_feat)
        prev_action = max(0, min(int(prev_action), self._num_actions - 1))
        one_hot = torch.zeros(conv_feat.shape[0], self._num_actions, device=conv_feat.device)
        one_hot.scatter_(
            1,
            torch.full(
                (conv_feat.shape[0], 1),
                prev_action,
                dtype=torch.long,
                device=conv_feat.device,
            ),
            1.0,
        )
        prev_r = torch.full(
            (conv_feat.shape[0], 1),
            prev_reward,
            dtype=conv_feat.dtype,
            device=conv_feat.device,
        )
        lstm_in = torch.cat([conv_feat, one_hot, prev_r], dim=1).unsqueeze(1)
        out, (h_new, c_new) = self.lstm(lstm_in, (h, c))
        q = self.q_head(out.squeeze(1))
        return q, h_new, c_new

    def __call__(self, state, prev_action=0, prev_reward=0.0):
        prev_action = int(prev_action) if prev_action is not None else 0
        prev_reward = float(prev_reward) if prev_reward is not None else 0.0
        state = torch.as_tensor(state, dtype=torch.float32)
        with torch.inference_mode():
            q, h_new, c_new = self.forward(
                state,
                prev_action=prev_action,
                prev_reward=prev_reward,
                h=self._h,
                c=self._c,
            )
            self._h, self._c = h_new, c_new
            action = q.argmax(dim=-1).squeeze(0)
        return int(action.item())
