import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from rl.backbone import BackboneSpec, HeadSpec, build_backbone, build_mlp_head
from rl.policy_backbone_utils import (
    ensure_env_spaces,
    init_linear,
    obs_space_from_env_conf,
)


@dataclass
class ActorPolicySpec:
    backbone: BackboneSpec
    head: HeadSpec
    action_squash: bool = True
    param_scale: float = 0.5


class ActorBackbonePolicy(nn.Module):
    def __init__(self, env_conf, spec: ActorPolicySpec):
        super().__init__()
        ensure_env_spaces(env_conf)
        obs_dim = int(obs_space_from_env_conf(env_conf).shape[0])
        act_dim = int(env_conf.action_space.shape[0])
        self._action_squash = bool(spec.action_squash)
        self._const_scale = float(spec.param_scale)
        self.backbone, feat_dim = build_backbone(spec.backbone, obs_dim)
        self.head = build_mlp_head(spec.head, feat_dim, act_dim)
        init_linear(self.backbone)
        init_linear(self.head)
        self._cache_flat_params_init()

    def _cache_flat_params_init(self):
        with torch.inference_mode():
            self._flat_params_init = torch.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()

    def num_params(self):
        return sum((p.numel() for p in self.parameters()))

    def get_params(self):
        with torch.inference_mode():
            flat = torch.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()
        return (flat - self._flat_params_init) / self._const_scale

    def set_params(self, flat_params):
        flat_params = np.asarray(flat_params, dtype=np.float32)
        assert flat_params.shape == self._flat_params_init.shape, (
            flat_params.shape,
            self._flat_params_init.shape,
        )
        flat = self._flat_params_init + flat_params * self._const_scale
        with torch.inference_mode():
            torch.nn.utils.vector_to_parameters(torch.from_numpy(flat).float(), list(self.parameters()))

    def clone(self):
        p = copy.deepcopy(self)
        if hasattr(p, "reset_state"):
            p.reset_state()
        return p

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(obs)
        out = self.head(feats)
        if self._action_squash:
            out = torch.tanh(out)
        return out

    def __call__(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)
        with torch.inference_mode():
            action = self.forward(state_t)
        return action.squeeze(0).detach().cpu().numpy()


class ActorBackbonePolicyFactory:
    def __init__(
        self,
        backbone: BackboneSpec,
        head: HeadSpec,
        *,
        action_squash: bool = True,
        param_scale: float = 0.5,
    ):
        self._spec = ActorPolicySpec(
            backbone=backbone,
            head=head,
            action_squash=action_squash,
            param_scale=param_scale,
        )

    def __call__(self, env_conf):
        return ActorBackbonePolicy(env_conf, self._spec)
