from dataclasses import dataclass

import torch
import torch.nn as nn

from policies.policy_mixin import PolicyParamsMixin
from rl.backbone import BackboneSpec, HeadSpec, build_backbone, build_mlp_head
from rl.core import env_contract as torchrl_env_contract
from rl.policy_backbone_utils import ensure_env_spaces, init_linear


@dataclass
class DiscreteActorPolicySpec:
    backbone: BackboneSpec
    head: HeadSpec
    param_scale: float = 0.5


class DiscreteActorBackbonePolicy(PolicyParamsMixin, nn.Module):
    def __init__(self, env_conf, spec: DiscreteActorPolicySpec):
        super().__init__()
        ensure_env_spaces(env_conf)
        self.problem_seed = env_conf.problem_seed
        self._env_conf = env_conf
        self._const_scale = float(spec.param_scale)
        io_contract = torchrl_env_contract.resolve_env_io_contract(env_conf, default_image_size=84)
        if io_contract.action.kind != "discrete":
            raise ValueError("DiscreteActorBackbonePolicy expects a discrete action space.")
        self._obs_contract = io_contract.observation
        obs_dim = 64 if self._obs_contract.mode == "pixels" else int(self._obs_contract.vector_dim or 1)
        backbone_name = torchrl_env_contract.resolve_backbone_name(spec.backbone.name, self._obs_contract)
        backbone_spec = BackboneSpec(
            name=backbone_name,
            hidden_sizes=tuple(spec.backbone.hidden_sizes),
            activation=spec.backbone.activation,
            layer_norm=bool(spec.backbone.layer_norm),
        )
        self.backbone, feat_dim = build_backbone(backbone_spec, obs_dim)
        self.head = build_mlp_head(spec.head, feat_dim, int(io_contract.action.dim))
        init_linear(self.backbone)
        init_linear(self.head)
        self._cache_flat_params_init()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        squeeze_batch_dim = False
        if self._obs_contract.mode == "pixels":
            from rl.core.pixel_transform import ensure_pixel_obs_format

            obs = ensure_pixel_obs_format(
                obs,
                channels=int(self._obs_contract.model_channels or 3),
                size=int(self._obs_contract.image_size or 84),
                scale_float_255=False,
            )
            if obs.ndim == 3:
                obs = obs.unsqueeze(0)
                squeeze_batch_dim = True
        elif obs.ndim == 1:
            obs = obs.unsqueeze(0)
            squeeze_batch_dim = True
        elif obs.ndim > 2:
            obs = obs.reshape(obs.shape[0], -1)
        feats = self.backbone(obs)
        logits = self.head(feats)
        if squeeze_batch_dim:
            logits = logits.squeeze(0)
        return logits

    def __call__(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        with torch.inference_mode():
            logits = self.forward(state_t)
        if logits.ndim == 1:
            return int(logits.argmax(dim=-1).item())
        action = logits.argmax(dim=-1)
        if action.numel() == 1:
            return int(action.reshape(()).item())
        return action.detach().cpu().numpy()


class DiscreteActorBackbonePolicyFactory:
    def __init__(self, backbone: BackboneSpec, head: HeadSpec, *, param_scale: float = 0.5):
        self._spec = DiscreteActorPolicySpec(backbone=backbone, head=head, param_scale=param_scale)

    def __call__(self, env_conf):
        return DiscreteActorBackbonePolicy(env_conf, self._spec)
