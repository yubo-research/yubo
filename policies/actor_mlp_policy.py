import torch
import torch.nn as nn
from torch.distributions import Normal

from policies.env_utils import get_obs_act_dims
from policies.policy_mixin import PolicyParamsMixin
from rl.actor_critic import gaussian_policy_normal_from_obs
from rl.backbone import BackboneSpec, HeadSpec, build_backbone, build_mlp_head, init_linear_layers
from rl.math_utils import tanh_gaussian_action_log_prob_entropy


class ActorMLPPolicyFactory:
    def __init__(
        self,
        hidden_sizes: tuple[int, ...],
        *,
        log_std_init: float = 0.0,
    ):
        self._hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self._log_std_init = float(log_std_init)

    def __call__(self, env_conf):
        return ActorMLPPolicy(
            env_conf,
            self._hidden_sizes,
            log_std_init=self._log_std_init,
        )


class ActorMLPPolicy(PolicyParamsMixin, nn.Module):
    def __init__(
        self,
        env_conf,
        hidden_sizes: tuple[int, ...],
        *,
        log_std_init: float = 0.0,
    ):
        super().__init__()
        self.problem_seed = env_conf.problem_seed
        obs_dim, act_dim = get_obs_act_dims(env_conf)
        obs_dim = int(obs_dim)
        act_dim = int(act_dim)
        self._last_log_prob: torch.Tensor | None = None

        backbone_spec = BackboneSpec(
            name="mlp",
            hidden_sizes=tuple(int(h) for h in hidden_sizes),
            activation="silu",
            layer_norm=True,
        )
        actor_head_spec = HeadSpec()

        self.actor_backbone, actor_feat_dim = build_backbone(backbone_spec, obs_dim)
        self.actor_head = build_mlp_head(actor_head_spec, actor_feat_dim, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

        init_linear_layers(self.actor_backbone, gain=0.5)
        init_linear_layers(self.actor_head, gain=0.5)
        self._const_scale = 0.5
        self._cache_flat_params_init()

    def _distribution(self, obs: torch.Tensor) -> Normal:
        return gaussian_policy_normal_from_obs(self.actor_backbone, self.actor_head, self.log_std, obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self._distribution(obs)
        return torch.tanh(dist.mean)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._distribution(obs)
        action, log_prob, entropy = tanh_gaussian_action_log_prob_entropy(dist, action)
        self._last_log_prob = log_prob
        return action, log_prob, entropy

    def last_log_probs(self) -> torch.Tensor | None:
        return self._last_log_prob

    def __call__(self, state):  # type: ignore[override]
        device = next(self.parameters()).device
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.inference_mode():
            action, _log_prob, _entropy = self.get_action_and_value(state)
        return action.detach().cpu().numpy()
