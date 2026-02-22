import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from problems.normalizer import Normalizer
from problems.policy_mixin import PolicyParamsMixin
from rl.algos.backends.torchrl.common import env_contract as torchrl_env_contract
from rl.backbone import BackboneSpec, HeadSpec, build_backbone, build_mlp_head
from rl.shared_gaussian_actor import SharedGaussianActorModule, get_gaussian_actor_spec


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
            self._flat_params_init = torch.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

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


@dataclass
class DiscreteActorPolicySpec:
    backbone: BackboneSpec
    head: HeadSpec
    param_scale: float = 0.5


class DiscreteActorBackbonePolicy(PolicyParamsMixin, nn.Module):
    def __init__(self, env_conf, spec: DiscreteActorPolicySpec):
        super().__init__()
        if env_conf.gym_conf is None:
            raise ValueError("DiscreteActorBackbonePolicy expects a gym env_conf.")
        env_conf.ensure_spaces()

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

        _init_linear(self.backbone)
        _init_linear(self.head)

        with torch.inference_mode():
            self._flat_params_init = np.concatenate([p.data.detach().cpu().numpy().reshape(-1) for p in self.parameters()])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        squeeze_batch_dim = False
        if self._obs_contract.mode == "pixels":
            from rl.algos.backends.torchrl.common.pixel_transform import (
                ensure_pixel_obs_format,
            )

            obs = ensure_pixel_obs_format(
                obs,
                channels=int(self._obs_contract.model_channels or 3),
                size=int(self._obs_contract.image_size or 84),
                scale_float_255=False,
            )
            if obs.ndim == 3:
                obs = obs.unsqueeze(0)
                squeeze_batch_dim = True
        else:
            if obs.ndim == 1:
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
    def __init__(
        self,
        backbone: BackboneSpec,
        head: HeadSpec,
        *,
        param_scale: float = 0.5,
    ):
        self._spec = DiscreteActorPolicySpec(
            backbone=backbone,
            head=head,
            param_scale=param_scale,
        )

    def __call__(self, env_conf):
        return DiscreteActorBackbonePolicy(env_conf, self._spec)


class AtariMLP16DiscretePolicy(DiscreteActorBackbonePolicy):
    def __init__(self, env_conf):
        super().__init__(
            env_conf,
            DiscreteActorPolicySpec(
                backbone=BackboneSpec(
                    name="mlp",
                    hidden_sizes=(16, 16),
                    activation="relu",
                    layer_norm=False,
                ),
                head=HeadSpec(
                    hidden_sizes=(16, 16),
                    activation="relu",
                ),
                param_scale=0.5,
            ),
        )


class GaussianActorBackbonePolicy(PolicyParamsMixin, nn.Module):
    """Gaussian policy using SharedGaussianActorModule. BO-compatible (get_params/set_params)."""

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
        self.problem_seed = env_conf.problem_seed
        self._env_conf = env_conf
        self._deterministic_eval = bool(deterministic_eval)
        mode = str(squash_mode).strip().lower()
        if mode not in {"clip", "tanh_clip"}:
            raise ValueError(f"Unsupported squash_mode '{squash_mode}'. Expected: clip, tanh_clip.")
        self._squash_mode = mode

        num_state = int(env_conf.gym_conf.state_space.shape[0])
        num_action = int(env_conf.action_space.shape[0])
        self._normalizer = Normalizer(shape=(num_state,))
        self._clamp = env_conf.gym_conf is not None

        self.actor = SharedGaussianActorModule(
            num_state,
            num_action,
            *get_gaussian_actor_spec(variant),
            init_log_std=init_log_std,
        )
        self._const_scale = 0.5
        self._cache_flat_params_init()

    def _cache_flat_params_init(self):
        with torch.inference_mode():
            self._flat_params_init = np.concatenate([p.data.detach().cpu().numpy().reshape(-1) for p in self.parameters()])

    def _normalize(self, state):
        state = np.asarray(state, dtype=np.float32)
        self._normalizer.update(state)
        mean, std = self._normalizer.mean_and_std()
        std = np.where(std == 0, 1.0, std)
        return (state - mean) / std

    def _postprocess_action(self, action_t: torch.Tensor) -> torch.Tensor:
        if self._squash_mode == "tanh_clip":
            action_t = torch.tanh(action_t)
        if self._clamp:
            action_t = torch.clamp(action_t, -1, 1)
        return action_t

    def __call__(self, state):
        state = self._normalize(state)
        state_t = torch.as_tensor(state, dtype=torch.float32)
        with torch.inference_mode():
            raw_action_t, _, _ = self.actor.sample_action(state_t, deterministic=self._deterministic_eval)
            action_t = self._postprocess_action(raw_action_t)
        return action_t.detach().cpu().numpy()


class GaussianActorBackbonePolicyFactory:
    """Factory for GaussianActorBackbonePolicy. Registry variants: rl-gauss, rl-gauss-tanh, rl-gauss-small."""

    def __init__(
        self,
        variant: str = "rl-gauss-tanh",
        *,
        deterministic_eval: bool = True,
        squash_mode: str = "clip",
        init_log_std: float = -0.5,
    ):
        self._variant = str(variant)
        self._deterministic_eval = bool(deterministic_eval)
        self._squash_mode = str(squash_mode)
        self._init_log_std = float(init_log_std)

    def __call__(self, env_conf):
        return GaussianActorBackbonePolicy(
            env_conf,
            variant=self._variant,
            deterministic_eval=self._deterministic_eval,
            squash_mode=self._squash_mode,
            init_log_std=self._init_log_std,
        )
