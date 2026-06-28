import torch
import torch.nn as nn
from torch.distributions import Normal

from policies.env_utils import get_obs_act_dims
from policies.policy_mixin import PolicyParamsMixin
from rl.actor_critic import gaussian_policy_normal_from_obs
from rl.backbone import (
    BackboneSpec,
    HeadSpec,
    build_backbone,
    build_mlp_head,
    init_linear_layers,
)
from rl.math_utils import tanh_gaussian_action_log_prob_entropy


class ActorCriticMLPPolicyFactory:
    def __init__(
        self,
        hidden_sizes: tuple[int, ...],
        *,
        activation: str = "silu",
        layer_norm: bool = True,
        share_backbone: bool = True,
        log_std_init: float = 0.0,
        squash_action: bool = True,
        deterministic_call: bool = False,
    ):
        self._hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self._activation = str(activation)
        self._layer_norm = bool(layer_norm)
        self._share_backbone = bool(share_backbone)
        self._log_std_init = float(log_std_init)
        self._squash_action = bool(squash_action)
        self._deterministic_call = bool(deterministic_call)

    def __call__(self, env_conf):
        return ActorCriticMLPPolicy(
            env_conf,
            self._hidden_sizes,
            activation=self._activation,
            layer_norm=self._layer_norm,
            share_backbone=self._share_backbone,
            log_std_init=self._log_std_init,
            squash_action=self._squash_action,
            deterministic_call=self._deterministic_call,
        )


class ActorCriticMLPPolicy(PolicyParamsMixin, nn.Module):
    """Actor-critic policy for PPO with Gaussian actions and tanh squashing.

    Unlike ActorCritic from rl/actor_critic.py, this class:
    - Uses env_conf factory pattern for construction
    - Provides numpy __call__ interface for trajectory collection
    - Caches last_log_probs/last_values for PPO integration
    - Inherits PolicyParamsMixin for get_params/set_params/clone
    """

    def __init__(
        self,
        env_conf,
        hidden_sizes: tuple[int, ...],
        *,
        activation: str = "silu",
        layer_norm: bool = True,
        share_backbone: bool = True,
        log_std_init: float = 0.0,
        squash_action: bool = True,
        deterministic_call: bool = False,
    ):
        super().__init__()
        self.problem_seed = env_conf.problem_seed
        obs_dim, act_dim = get_obs_act_dims(env_conf)
        obs_dim = int(obs_dim)
        act_dim = int(act_dim)
        self._last_log_prob: torch.Tensor | None = None
        self._last_value: torch.Tensor | None = None
        self._squash_action = bool(squash_action)
        self._deterministic_call = bool(deterministic_call)

        backbone_spec = BackboneSpec(
            name="mlp",
            hidden_sizes=tuple(int(h) for h in hidden_sizes),
            activation=str(activation),
            layer_norm=bool(layer_norm),
        )
        actor_head_spec = HeadSpec(activation=str(activation))
        critic_head_spec = HeadSpec(activation=str(activation))

        if share_backbone:
            backbone, feat_dim = build_backbone(backbone_spec, obs_dim)
            self.actor_backbone = backbone
            self.critic_backbone = backbone
            actor_feat_dim = feat_dim
            critic_feat_dim = feat_dim
        else:
            self.actor_backbone, actor_feat_dim = build_backbone(backbone_spec, obs_dim)
            self.critic_backbone, critic_feat_dim = build_backbone(backbone_spec, obs_dim)

        self.actor_head = build_mlp_head(actor_head_spec, actor_feat_dim, act_dim)
        self.critic_head = build_mlp_head(critic_head_spec, critic_feat_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

        init_linear_layers(self.actor_backbone, gain=0.5)
        if not share_backbone:
            init_linear_layers(self.critic_backbone, gain=0.5)
        init_linear_layers(self.actor_head, gain=0.5)
        init_linear_layers(self.critic_head, gain=0.5)
        self._const_scale = 0.5
        self._cache_flat_params_init()

    def _distribution(self, obs: torch.Tensor) -> Normal:
        return gaussian_policy_normal_from_obs(self.actor_backbone, self.actor_head, self.log_std, obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self._distribution(obs)
        if self._squash_action:
            return torch.tanh(dist.mean)
        return torch.clamp(dist.mean, -1.0, 1.0)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        feats = self.critic_backbone(obs)
        return self.critic_head(feats).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._distribution(obs)
        if self._deterministic_call and action is None:
            action = torch.tanh(dist.mean) if self._squash_action else torch.clamp(dist.mean, -1.0, 1.0)
            entropy = dist.entropy().sum(-1)
            log_prob = torch.zeros(action.shape[:-1], dtype=action.dtype, device=action.device)
        elif self._squash_action:
            action, log_prob, entropy = tanh_gaussian_action_log_prob_entropy(dist, action)
        else:
            if action is None:
                action = torch.clamp(dist.rsample(), -1.0, 1.0)
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        value = self.get_value(obs)
        self._last_log_prob = log_prob
        self._last_value = value
        return action, log_prob, entropy, value

    def last_log_probs(self) -> torch.Tensor | None:
        return self._last_log_prob

    def last_values(self) -> torch.Tensor | None:
        return self._last_value

    def __call__(self, state):  # type: ignore[override]
        device = next(self.parameters()).device
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.inference_mode():
            action, _log_prob, _entropy, _value = self.get_action_and_value(state)
        return action.detach().cpu().numpy()
