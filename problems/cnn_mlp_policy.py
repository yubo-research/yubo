"""CNN + MLP policy for pixel-based observations (e.g. dm_control from_pixels)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from problems.policy_mixin import PolicyParamsMixin


def _tiny_atari_cnn_encoder(in_channels: int = 4) -> tuple[nn.Module, int]:
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


def _nature_cnn_encoder(in_channels: int = 3, latent_dim: int = 256) -> tuple[nn.Module, int]:
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
    # Output dim: 64 * 1 * 1 = 64
    with torch.inference_mode():
        dummy = torch.zeros(1, in_channels, 84, 84)
        out_dim = encoder(dummy).shape[-1]
    if latent_dim != out_dim:
        encoder = nn.Sequential(encoder, nn.Linear(out_dim, latent_dim), nn.ReLU())
        out_dim = latent_dim
    return encoder, out_dim


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

        obs_space = env_conf.gym_conf.state_space
        assert hasattr(obs_space, "shape"), "Pixel policy expects Box observation space"
        shape = obs_space.shape
        assert len(shape) == 3, f"Expected (H,W,C), got {shape}"
        h, w, c = int(shape[0]), int(shape[1]), int(shape[2])
        assert h == 84 and w == 84, f"Expected 84x84 pixels, got {h}x{w}"
        num_action = int(env_conf.action_space.shape[0])

        self.encoder, feat_dim = _nature_cnn_encoder(in_channels=c, latent_dim=cnn_latent_dim)
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
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., H, W, C) -> (..., C, H, W)
        if x.shape[-1] == 3:
            x = x.permute(*range(x.dim() - 3), -1, -3, -2)
        feats = self.encoder(x)
        return self.head(feats)

    def __call__(self, state):
        # state: (H, W, C) numpy, float in [0,1] from collect_trajectory normalization
        state = torch.as_tensor(state, dtype=torch.float32)
        if state.dim() == 3:
            state = state.unsqueeze(0)
        with torch.inference_mode():
            action = self.forward(state)
        return action.squeeze(0).detach().cpu().numpy()


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

        obs_space = env_conf.gym_conf.state_space
        shape = obs_space.shape
        # Atari: (4, 84, 84), (84, 84, 4), or (4, 84, 84, 1) from FrameStack + grayscale
        if len(shape) == 4 and shape[-1] == 1:
            in_channels = int(shape[0])
        elif len(shape) == 3:
            if shape[0] == 4:
                in_channels = 4
            else:
                in_channels = int(shape[-1])
        else:
            raise ValueError(f"Expected 3D or 4D obs (4,84,84), (84,84,4), (4,84,84,1), got {shape}")
        assert 84 in shape[:3], f"Expected 84x84, got {shape}"

        num_actions = int(env_conf.action_space.n)
        if str(variant).lower() == "small":
            self.encoder, feat_dim = _tiny_atari_cnn_encoder(in_channels=in_channels)
            dims = [feat_dim] + list(hidden_sizes) + [num_actions] if hidden_sizes else [feat_dim, 24, num_actions]
        else:
            self.encoder, feat_dim = _nature_cnn_encoder(in_channels=in_channels, latent_dim=cnn_latent_dim)
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
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C,H,W), (H,W,C), (4,H,W,1), or (N,4,H,W,1) -> (N,C,H,W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.unsqueeze(0)[..., 0]  # (4,84,84,1) -> (1,4,84,84)
        elif x.dim() == 5 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.shape[1] in (3, 4) and x.shape[2] == 84:
            pass  # already (N,C,H,W)
        elif x.shape[-1] in (3, 4):
            x = x.permute(0, 3, 1, 2)
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


class AtariGaussianPolicyFactory:
    """Factory for Atari policy with Gaussian head over action logits -> discrete via argmax."""

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
    """CNN + Gaussian head for Atari. Outputs mean over n_actions; discrete action = argmax(mean) or argmax(sample)."""

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

        obs_space = env_conf.gym_conf.state_space
        shape = obs_space.shape
        if len(shape) == 4 and shape[-1] == 1:
            in_channels = int(shape[0])
        elif len(shape) == 3:
            in_channels = int(shape[-1]) if shape[0] != 4 else 4
        else:
            raise ValueError(f"Expected 3D or 4D obs, got {shape}")
        num_actions = int(env_conf.action_space.n)

        if variant == "small":
            self.encoder, feat_dim = _tiny_atari_cnn_encoder(in_channels=in_channels)
        else:
            self.encoder, feat_dim = _nature_cnn_encoder(in_channels=in_channels, latent_dim=cnn_latent_dim)
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
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _to_chw(self, x: torch.Tensor) -> torch.Tensor:
        # Match AtariCNNPolicy: (C,H,W), (H,W,C), (4,H,W,1), (N,4,H,W,1) -> (N,C,H,W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.unsqueeze(0)[..., 0]  # (4,84,84,1) -> (1,4,84,84)
        elif x.dim() == 5 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.shape[1] in (3, 4) and x.shape[2] == 84:
            pass  # already (N,C,H,W)
        elif x.shape[-1] in (3, 4):
            x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._to_chw(x)
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

        obs_space = env_conf.gym_conf.state_space
        shape = obs_space.shape
        if len(shape) == 4 and shape[-1] == 1:
            in_channels = int(shape[0])
        elif len(shape) == 3:
            in_channels = int(shape[-1]) if shape[0] != 4 else 4
        else:
            raise ValueError(f"Expected 3D or 4D obs, got {shape}")
        num_actions = int(env_conf.action_space.n)

        if cnn_variant == "small":
            self.encoder, conv_dim = _tiny_atari_cnn_encoder(in_channels=in_channels)
        else:
            self.encoder, conv_dim = _nature_cnn_encoder(in_channels=in_channels, latent_dim=64)
        # LSTM input: proj(conv_feat) + one_hot(prev_action) + prev_reward
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

    def _to_chw(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.unsqueeze(0)[..., 0]
        elif x.dim() == 5 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.shape[1] in (3, 4) and x.shape[2] == 84:
            pass
        elif x.shape[-1] in (3, 4):
            x = x.permute(0, 3, 1, 2)
        return x

    def forward(
        self,
        x: torch.Tensor,
        prev_action: int = 0,
        prev_reward: float = 0.0,
        h=None,
        c=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._to_chw(x)
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
