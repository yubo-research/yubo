from __future__ import annotations

import torch
import torch.nn as nn

from policies.env_utils import get_obs_act_dims
from policies.policy_mixin import PolicyParamsMixin
from rl.backbone import BackboneSpec, HeadSpec, build_backbone, build_mlp_head, init_linear_layers


class MoEPolicyFactory:
    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (16, 16),
        *,
        num_experts: int = 4,
        router_hidden_sizes: tuple[int, ...] = (16,),
        top_k: int | None = 2,
        temperature: float = 1.0,
        use_layer_norm: bool = True,
        activation: str = "silu",
    ):
        self._hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self._num_experts = int(num_experts)
        self._router_hidden_sizes = tuple(int(h) for h in router_hidden_sizes)
        self._top_k = None if top_k is None else int(top_k)
        self._temperature = float(temperature)
        self._use_layer_norm = bool(use_layer_norm)
        self._activation = str(activation)

    def __call__(self, env_conf):
        return MoEPolicy(
            env_conf,
            hidden_sizes=self._hidden_sizes,
            num_experts=self._num_experts,
            router_hidden_sizes=self._router_hidden_sizes,
            top_k=self._top_k,
            temperature=self._temperature,
            use_layer_norm=self._use_layer_norm,
            activation=self._activation,
        )


class MoEPolicy(PolicyParamsMixin, nn.Module):
    """Mixture-of-experts policy with a learned router and expert MLPs."""

    def __init__(
        self,
        env_conf,
        *,
        hidden_sizes: tuple[int, ...] = (16, 16),
        num_experts: int = 4,
        router_hidden_sizes: tuple[int, ...] = (16,),
        top_k: int | None = 2,
        temperature: float = 1.0,
        use_layer_norm: bool = True,
        activation: str = "silu",
    ):
        super().__init__()
        self.problem_seed = env_conf.problem_seed
        obs_dim, act_dim = get_obs_act_dims(env_conf)
        obs_dim = int(obs_dim)
        act_dim = int(act_dim)
        if num_experts < 2:
            raise ValueError("MoEPolicy requires at least 2 experts.")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be >= 1 or None.")
        if top_k is not None and top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts.")
        self._num_experts = int(num_experts)
        self._top_k = top_k
        self._temperature = float(temperature)
        self._last_router_weights: torch.Tensor | None = None

        expert_spec = BackboneSpec(
            name="mlp",
            hidden_sizes=tuple(int(h) for h in hidden_sizes),
            activation=activation,
            layer_norm=use_layer_norm,
        )
        router_spec = BackboneSpec(
            name="mlp",
            hidden_sizes=tuple(int(h) for h in router_hidden_sizes),
            activation=activation,
            layer_norm=use_layer_norm,
        )
        self.router_backbone, router_dim = build_backbone(router_spec, obs_dim)
        self.router_head = build_mlp_head(HeadSpec(), router_dim, self._num_experts)

        self.expert_backbones = nn.ModuleList()
        self.expert_heads = nn.ModuleList()
        for _ in range(self._num_experts):
            backbone, feat_dim = build_backbone(expert_spec, obs_dim)
            head = build_mlp_head(HeadSpec(), feat_dim, act_dim)
            self.expert_backbones.append(backbone)
            self.expert_heads.append(head)

        init_linear_layers(self.router_backbone, gain=0.5)
        init_linear_layers(self.router_head, gain=0.5)
        for backbone in self.expert_backbones:
            init_linear_layers(backbone, gain=0.5)
        for head in self.expert_heads:
            init_linear_layers(head, gain=0.5)

        self._const_scale = 0.5
        self._cache_flat_params_init()

    def _router_weights(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.router_head(self.router_backbone(obs))
        if self._top_k is not None and self._top_k < self._num_experts:
            topk_idx = torch.topk(logits, k=self._top_k, dim=-1).indices
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(-1, topk_idx, True)
            logits = torch.where(mask, logits, torch.full_like(logits, -1e9))
        weights = torch.softmax(logits / max(self._temperature, 1e-6), dim=-1)
        return weights

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        weights = self._router_weights(obs)
        expert_actions = []
        for backbone, head in zip(self.expert_backbones, self.expert_heads):
            feats = backbone(obs)
            expert_actions.append(head(feats))
        stacked = torch.stack(expert_actions, dim=-2)
        action = torch.sum(stacked * weights.unsqueeze(-1), dim=-2)
        self._last_router_weights = weights
        return action

    def last_router_weights(self) -> torch.Tensor | None:
        return self._last_router_weights

    def __call__(self, state):  # type: ignore[override]
        device = next(self.parameters()).device
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.inference_mode():
            action = self.forward(state)
        return action.detach().cpu().numpy()
