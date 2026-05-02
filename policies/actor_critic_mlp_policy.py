import torch
import torch.nn as nn
from torch.distributions import Normal

from policies.env_utils import get_obs_act_dims
from policies.policy_mixin import PolicyParamsMixin
from rl.backbone import BackboneSpec, HeadSpec, build_backbone, build_mlp_head, init_linear_layers
from rl.math_utils import atanh


class ActorCriticMLPPolicyFactory:
    def __init__(
        self,
        hidden_sizes: tuple[int, ...],
        *,
        share_backbone: bool = True,
        log_std_init: float = 0.0,
    ):
        self._hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self._share_backbone = bool(share_backbone)
        self._log_std_init = float(log_std_init)

    def __call__(self, env_conf):
        return ActorCriticMLPPolicy(
            env_conf,
            self._hidden_sizes,
            share_backbone=self._share_backbone,
            log_std_init=self._log_std_init,
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
        share_backbone: bool = True,
        log_std_init: float = 0.0,
    ):
        super().__init__()
        self.problem_seed = env_conf.problem_seed
        obs_dim, act_dim = get_obs_act_dims(env_conf)
        obs_dim = int(obs_dim)
        act_dim = int(act_dim)
        self._last_log_prob: torch.Tensor | None = None
        self._last_value: torch.Tensor | None = None

        backbone_spec = BackboneSpec(
            name="mlp",
            hidden_sizes=tuple(int(h) for h in hidden_sizes),
            activation="silu",
            layer_norm=True,
        )
        actor_head_spec = HeadSpec()
        critic_head_spec = HeadSpec()

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
        feats = self.actor_backbone(obs)
        mean = self.actor_head(feats)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self._distribution(obs)
        return torch.tanh(dist.mean)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        feats = self.critic_backbone(obs)
        return self.critic_head(feats).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._distribution(obs)
        if action is None:
            u = dist.rsample()
            action = torch.tanh(u)
        else:
            u = atanh(action)
        log_prob = dist.log_prob(u) - torch.log(1.0 - action.pow(2) + 1e-06)
        log_prob = log_prob.sum(-1)
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
