import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from rl.backbone import BackboneSpec, HeadSpec, build_backbone, build_mlp_head


def _init_linear(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


@dataclass
class ActorPolicySpec:
    backbone: BackboneSpec
    head: HeadSpec
    action_squash: bool = True
    param_scale: float = 0.5


class ActorBackbonePolicy(nn.Module):
    def __init__(self, env_conf, spec: ActorPolicySpec):
        super().__init__()
        if env_conf.gym_conf is None:
            raise ValueError("ActorBackbonePolicy expects a gym env_conf.")
        env_conf.ensure_spaces()

        obs_dim = int(env_conf.gym_conf.state_space.shape[0])
        act_dim = int(env_conf.action_space.shape[0])
        self._action_squash = bool(spec.action_squash)
        self._const_scale = float(spec.param_scale)

        self.backbone, feat_dim = build_backbone(spec.backbone, obs_dim)
        self.head = build_mlp_head(spec.head, feat_dim, act_dim)

        _init_linear(self.backbone)
        _init_linear(self.head)

        self._cache_flat_params_init()

    def _cache_flat_params_init(self):
        with torch.inference_mode():
            self._flat_params_init = np.concatenate([p.data.detach().cpu().numpy().reshape(-1) for p in self.parameters()])

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_params(self):
        with torch.inference_mode():
            flat_params = np.concatenate([p.data.detach().cpu().numpy().reshape(-1) for p in self.parameters()])
        return (flat_params - self._flat_params_init) / self._const_scale

    def set_params(self, flat_params):
        flat_params = np.asarray(flat_params, dtype=np.float32)
        assert flat_params.shape == self._flat_params_init.shape, (
            flat_params.shape,
            self._flat_params_init.shape,
        )
        flat_params = self._flat_params_init + flat_params * self._const_scale
        with torch.inference_mode():
            idx = 0
            for p in self.parameters():
                shape = p.shape
                size = p.numel()
                p.copy_(torch.from_numpy(flat_params[idx : idx + size].reshape(shape)).float())
                idx += size

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
