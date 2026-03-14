from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from problems.activations import activation
from problems.normalizer import Normalizer
from problems.policy_mixin import PolicyParamsMixin

_GAUSSIAN_VARIANTS: dict[str, tuple[tuple[int, ...], str, bool]] = {
    "rl-gauss": ((64, 64), "silu", True),
    "rl-gauss-tanh": ((16, 16), "tanh", False),
    "rl-gauss-small": ((32, 16), "silu", True),
}


class GaussianPolicyFactory:
    def __init__(
        self,
        *,
        variant: str = "rl-gauss-tanh",
        deterministic_eval: bool = True,
        squash_mode: str = "clip",
        init_log_std: float = -0.5,
    ):
        self._variant = str(variant)
        self._deterministic_eval = bool(deterministic_eval)
        self._squash_mode = str(squash_mode)
        self._init_log_std = float(init_log_std)

    def __call__(self, env_conf):
        return GaussianPolicy(
            env_conf,
            variant=self._variant,
            deterministic_eval=self._deterministic_eval,
            squash_mode=self._squash_mode,
            init_log_std=self._init_log_std,
        )

    def to_rl_schema(self):
        return {
            "family": "gaussian_backbone",
            "variant": str(self._variant),
            "share_backbone": True,
            "log_std_init": float(self._init_log_std),
        }


class GaussianPolicy(PolicyParamsMixin, nn.Module):
    def __init__(
        self,
        env_conf,
        *,
        variant: str = "rl-gauss-tanh",
        deterministic_eval: bool = True,
        squash_mode: str = "clip",
        init_log_std: float = -0.5,
    ):
        super().__init__()
        if getattr(env_conf, "state_space", None) is None or getattr(env_conf, "action_space", None) is None:
            env_conf.ensure_spaces()
        self.problem_seed = env_conf.problem_seed
        self._env_conf = env_conf
        self._deterministic_eval = bool(deterministic_eval)
        mode = str(squash_mode).strip().lower()
        if mode not in {"clip", "tanh_clip"}:
            raise ValueError(f"Unsupported squash_mode '{squash_mode}'. Expected one of: clip, tanh_clip.")
        self._squash_mode = mode
        key = str(variant).strip().lower()
        if key not in _GAUSSIAN_VARIANTS:
            raise ValueError(f"Unknown Gaussian policy variant '{variant}'. Available: {sorted(_GAUSSIAN_VARIANTS)}")
        hidden_sizes, act_name, use_layer_norm = _GAUSSIAN_VARIANTS[key]
        num_state = int(env_conf.state_space.shape[0])
        num_action = int(env_conf.action_space.shape[0])
        self._normalizer = Normalizer(shape=(num_state,))
        self._clamp = env_conf.gym_conf is not None
        act = activation(act_name)
        dims = [num_state] + list(hidden_sizes) + [num_action]
        layers: list[nn.Module] = []
        if bool(use_layer_norm):
            layers.append(nn.LayerNorm(num_state, elementwise_affine=True))
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.mean_net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.full((num_action,), float(init_log_std), dtype=torch.float32))
        self._const_scale = 0.5
        self._init_params()
        with torch.inference_mode():
            self._flat_params_init = np.concatenate([p.data.detach().cpu().numpy().reshape(-1) for p in self.parameters()])

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _normalize(self, state):
        state = np.asarray(state, dtype=np.float32)
        self._normalizer.update(state)
        mean, std = self._normalizer.mean_and_std()
        std = np.where(std == 0, 1.0, std)
        return (state - mean) / std

    def _postprocess_action(self, action_t: torch.Tensor) -> torch.Tensor:
        out = torch.tanh(action_t) if self._squash_mode == "tanh_clip" else action_t
        if self._clamp:
            return out.clamp(-1, 1)
        return out

    def __call__(self, state):
        state = self._normalize(state)
        device = next(self.parameters()).device
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.inference_mode():
            mean = self.mean_net(state_t)
            if self._deterministic_eval:
                raw_action = mean
            else:
                std = torch.exp(torch.clamp(self.log_std, -20.0, 2.0))
                raw_action = torch.distributions.Normal(mean, std).rsample()
            action = self._postprocess_action(raw_action)
        return action.detach().cpu().numpy()
