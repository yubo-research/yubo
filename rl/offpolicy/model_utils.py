from __future__ import annotations

import copy
import dataclasses
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from rl.backbone import (
    ActorCriticSpec,
    BackboneSpec,
    HeadSpec,
    build_backbone,
    build_mlp_head,
)
from rl.gac_actor import GACActorNet

from .models import QNet, QNetPixel
from .runtime_utils import ObsScaler


class ActorNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        obs_scaler: ObsScaler,
        *,
        act_dim: int,
    ):
        super().__init__()
        self.backbone, self.head = backbone, head
        self.obs_scaler = obs_scaler
        self.act_dim = int(act_dim)

    def _mean_log_std(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(self.obs_scaler(obs))
        out = self.head(feats)
        mean = out[..., : self.act_dim]
        log_std = out[..., self.act_dim :].clamp(-5.0, 2.0)
        return (mean, log_std)

    def mean_log_std(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._mean_log_std(obs)

    def log_prob_from_action(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, log_std = self._mean_log_std(obs)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        action_clamped = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = torch.atanh(action_clamped)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - action_clamped.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)

    def sample(self, obs: torch.Tensor, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self._mean_log_std(obs)
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros(action.shape[0], dtype=action.dtype, device=action.device)
            return (action, log_prob)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        return (action, log_prob.sum(dim=-1))

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self._mean_log_std(obs)
        return torch.tanh(mean)


@dataclasses.dataclass
class OffPolicyModules:
    actor_backbone: nn.Module
    actor_head: nn.Module | None
    actor_direction_head: nn.Module | None
    actor_concentration_head: nn.Module | None
    actor_scale_head: nn.Module | None
    q1_backbone: nn.Module
    q1_head: nn.Module
    q2_backbone: nn.Module
    q2_head: nn.Module
    obs_scaler: ObsScaler
    actor: ActorNet | GACActorNet
    q1: nn.Module
    q2: nn.Module
    q1_target: nn.Module
    q2_target: nn.Module


@dataclasses.dataclass
class OffPolicyOptimizers:
    actor_optimizer: optim.Optimizer
    critic_optimizer: optim.Optimizer


@dataclasses.dataclass
class _QBundle:
    q1_backbone: nn.Module
    q1_head: nn.Module
    q2_backbone: nn.Module
    q2_head: nn.Module
    q1: nn.Module
    q2: nn.Module


def _resolve_backbone_name(backbone_name: Any, obs_spec: Any) -> str:
    if getattr(obs_spec, "mode", "vector") != "pixels":
        return str(backbone_name)
    channels = int(getattr(obs_spec, "channels", 3) or 3)
    key = str(backbone_name).strip().lower()
    if key in {"mlp", "nature_cnn"} and channels == 4:
        return "nature_cnn_atari"
    if key in {"mlp", "nature_cnn_atari"} and channels != 4:
        return "nature_cnn"
    return str(backbone_name)


def _build_arch(config: Any, obs_spec: Any) -> ActorCriticSpec:
    actor_backbone_name = _resolve_backbone_name(config.backbone_name, obs_spec)
    critic_backbone_source = config.backbone_name if getattr(config, "critic_backbone_name", None) is None else config.critic_backbone_name
    critic_backbone_name = _resolve_backbone_name(critic_backbone_source, obs_spec)
    return ActorCriticSpec.from_config(
        config,
        actor_backbone_name=actor_backbone_name,
        critic_backbone_name=critic_backbone_name,
    )


def _build_q_modules(
    obs_spec: Any,
    env: Any,
    critic_backbone_spec: BackboneSpec,
    critic_head_spec: HeadSpec,
    obs_scaler: ObsScaler,
) -> _QBundle:
    obs_dim = int(obs_spec.vector_dim or 64)
    if obs_spec.mode == "pixels":
        q1_backbone, q1_feat_dim = build_backbone(critic_backbone_spec, input_dim=obs_dim)
        q2_backbone, q2_feat_dim = build_backbone(critic_backbone_spec, input_dim=obs_dim)
        q1_head = build_mlp_head(critic_head_spec, input_dim=q1_feat_dim + int(env.act_dim), output_dim=1)
        q2_head = build_mlp_head(critic_head_spec, input_dim=q2_feat_dim + int(env.act_dim), output_dim=1)
        q1 = QNetPixel(q1_backbone, q1_head, obs_scaler)
        q2 = QNetPixel(q2_backbone, q2_head, obs_scaler)
        return _QBundle(
            q1_backbone=q1_backbone,
            q1_head=q1_head,
            q2_backbone=q2_backbone,
            q2_head=q2_head,
            q1=q1,
            q2=q2,
        )
    q_input_dim = int(obs_spec.vector_dim or 1) + int(env.act_dim)
    q1_backbone, q1_feat_dim = build_backbone(critic_backbone_spec, input_dim=q_input_dim)
    q2_backbone, q2_feat_dim = build_backbone(critic_backbone_spec, input_dim=q_input_dim)
    q1_head = build_mlp_head(critic_head_spec, input_dim=q1_feat_dim, output_dim=1)
    q2_head = build_mlp_head(critic_head_spec, input_dim=q2_feat_dim, output_dim=1)
    q1 = QNet(q1_backbone, q1_head, obs_scaler)
    q2 = QNet(q2_backbone, q2_head, obs_scaler)
    return _QBundle(
        q1_backbone=q1_backbone,
        q1_head=q1_head,
        q2_backbone=q2_backbone,
        q2_head=q2_head,
        q1=q1,
        q2=q2,
    )


def _build_gac_actor(
    arch: ActorCriticSpec,
    obs_dim: int,
    act_dim: int,
    obs_scaler: ObsScaler,
    *,
    action_radius: float = 2.5,
    use_adaptive_scale: bool = False,
) -> tuple[GACActorNet, nn.Module, nn.Module, nn.Module | None]:
    actor_backbone, actor_feat_dim = build_backbone(arch.actor.backbone, input_dim=obs_dim)
    direction_head = build_mlp_head(arch.actor.head, input_dim=actor_feat_dim, output_dim=act_dim)
    concentration_head = nn.Sequential(
        nn.Linear(actor_feat_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    # Init output ~0 so κ starts near 0 (w=0.5, balanced exploration)
    nn.init.zeros_(concentration_head[-1].bias)
    nn.init.orthogonal_(concentration_head[-1].weight, gain=0.01)
    scale_head = nn.Linear(actor_feat_dim, act_dim) if use_adaptive_scale else None
    actor = GACActorNet(
        backbone=actor_backbone,
        direction_head=direction_head,
        concentration_head=concentration_head,
        obs_scaler=obs_scaler,
        act_dim=act_dim,
        action_radius=action_radius,
        use_adaptive_scale=use_adaptive_scale,
        scale_head=scale_head,
    )
    return actor, direction_head, concentration_head, scale_head


def build_modules(config: Any, env: Any, obs_spec: Any, *, device: torch.device) -> OffPolicyModules:
    obs_scaler = ObsScaler(env.obs_lb, env.obs_width).to(device)
    arch = _build_arch(config, obs_spec)
    obs_dim = int(obs_spec.vector_dim or 64)
    act_dim = int(env.act_dim)
    use_gac = str(getattr(config, "actor_type", "gaussian")).strip().lower() == "gac"

    if use_gac:
        action_radius = float(getattr(config, "gac_action_radius", 2.5))
        use_adaptive_scale = bool(getattr(config, "gac_adaptive_scale", False))
        actor, direction_head, concentration_head, scale_head = _build_gac_actor(
            arch, obs_dim, act_dim, obs_scaler, action_radius=action_radius, use_adaptive_scale=use_adaptive_scale
        )
        actor_head = None
    else:
        actor_backbone, actor_feat_dim = build_backbone(arch.actor.backbone, input_dim=obs_dim)
        actor_head = build_mlp_head(arch.actor.head, input_dim=actor_feat_dim, output_dim=2 * act_dim)
        actor = ActorNet(actor_backbone, actor_head, obs_scaler, act_dim=act_dim)
        direction_head = concentration_head = scale_head = None

    q_bundle = _build_q_modules(obs_spec, env, arch.critic.backbone, arch.critic.head, obs_scaler)
    actor.to(device)
    q_bundle.q1.to(device)
    q_bundle.q2.to(device)
    q1_target = copy.deepcopy(q_bundle.q1).to(device).eval()
    q2_target = copy.deepcopy(q_bundle.q2).to(device).eval()
    for param in q1_target.parameters():
        param.requires_grad_(False)
    for param in q2_target.parameters():
        param.requires_grad_(False)

    if use_gac:
        actor_backbone = actor.backbone
        actor_param_count = (
            sum((p.numel() for p in actor_backbone.parameters()))
            + sum((p.numel() for p in direction_head.parameters()))
            + sum((p.numel() for p in concentration_head.parameters()))
            + (sum((p.numel() for p in scale_head.parameters())) if scale_head else 0)
        )
    else:
        actor_backbone = actor.backbone
        actor_param_count = sum((p.numel() for p in actor_backbone.parameters())) + sum((p.numel() for p in actor_head.parameters()))

    if getattr(config, "theta_dim", None) is not None:
        assert int(actor_param_count) == int(config.theta_dim), (actor_param_count, config.theta_dim)

    return OffPolicyModules(
        actor_backbone=actor_backbone,
        actor_head=actor_head,
        actor_direction_head=direction_head,
        actor_concentration_head=concentration_head,
        actor_scale_head=scale_head,
        q1_backbone=q_bundle.q1_backbone,
        q1_head=q_bundle.q1_head,
        q2_backbone=q_bundle.q2_backbone,
        q2_head=q_bundle.q2_head,
        obs_scaler=obs_scaler,
        actor=actor,
        q1=q_bundle.q1,
        q2=q_bundle.q2,
        q1_target=q1_target,
        q2_target=q2_target,
    )


def build_optimizers(config: Any, modules: OffPolicyModules) -> OffPolicyOptimizers:
    actor_params = list(modules.actor_backbone.parameters())
    if modules.actor_head is not None:
        actor_params += list(modules.actor_head.parameters())
    else:
        actor_params += list(modules.actor_direction_head.parameters())
        actor_params += list(modules.actor_concentration_head.parameters())
        if modules.actor_scale_head is not None:
            actor_params += list(modules.actor_scale_head.parameters())
    critic_params = list(modules.q1.parameters()) + list(modules.q2.parameters())
    return OffPolicyOptimizers(
        actor_optimizer=optim.AdamW(actor_params, lr=float(config.learning_rate_actor), weight_decay=0.0),
        critic_optimizer=optim.AdamW(critic_params, lr=float(config.learning_rate_critic), weight_decay=0.0),
    )


def capture_actor_state(modules: Any) -> dict[str, Any]:
    out = {
        "backbone": {key: value.detach().clone() for key, value in modules.actor_backbone.state_dict().items()},
        "obs_scaler": {key: value.detach().clone() for key, value in modules.obs_scaler.state_dict().items()},
    }
    if modules.actor_head is not None:
        out["head"] = {key: value.detach().clone() for key, value in modules.actor_head.state_dict().items()}
    else:
        out["direction_head"] = {key: value.detach().clone() for key, value in modules.actor_direction_head.state_dict().items()}
        out["concentration_head"] = {key: value.detach().clone() for key, value in modules.actor_concentration_head.state_dict().items()}
        if modules.actor_scale_head is not None:
            out["scale_head"] = {key: value.detach().clone() for key, value in modules.actor_scale_head.state_dict().items()}
    return out


def restore_actor_state(modules: Any, snapshot: dict[str, Any]) -> None:
    modules.actor_backbone.load_state_dict(snapshot["backbone"])
    if "head" in snapshot:
        modules.actor_head.load_state_dict(snapshot["head"])
    else:
        modules.actor_direction_head.load_state_dict(snapshot["direction_head"])
        modules.actor_concentration_head.load_state_dict(snapshot["concentration_head"])
        if "scale_head" in snapshot and modules.actor_scale_head is not None:
            modules.actor_scale_head.load_state_dict(snapshot["scale_head"])
    if "obs_scaler" in snapshot:
        modules.obs_scaler.load_state_dict(snapshot["obs_scaler"])


@contextmanager
def use_actor_state(modules: Any, snapshot: dict[str, Any]):
    previous = capture_actor_state(modules)
    restore_actor_state(modules, snapshot)
    try:
        yield
    finally:
        restore_actor_state(modules, previous)


__all__ = [
    "ActorNet",
    "QNet",
    "QNetPixel",
    "OffPolicyModules",
    "OffPolicyOptimizers",
    "build_modules",
    "build_optimizers",
    "capture_actor_state",
    "restore_actor_state",
    "use_actor_state",
]
