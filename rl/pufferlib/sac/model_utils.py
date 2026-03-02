"""Model and optimization helpers for native Puffer SAC."""

from __future__ import annotations

import copy
import dataclasses
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from rl.backbone import BackboneSpec, HeadSpec, build_backbone, build_mlp_head
from rl.core.sac_update import (
    SACUpdateBatch,
    SACUpdateHyperParams,
    SACUpdateModules,
    SACUpdateOptimizers,
    sac_update_step,
)

from .config import SACConfig
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
        self.backbone = backbone
        self.head = head
        self.obs_scaler = obs_scaler
        self.act_dim = int(act_dim)

    def _mean_log_std(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(self.obs_scaler(obs))
        out = self.head(feats)
        mean = out[..., : self.act_dim]
        log_std = out[..., self.act_dim :].clamp(-5.0, 2.0)
        return mean, log_std

    def sample(self, obs: torch.Tensor, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self._mean_log_std(obs)
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros(action.shape[0], dtype=action.dtype, device=action.device)
            return action, log_prob

        std = torch.exp(log_std)
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self._mean_log_std(obs)
        return torch.tanh(mean)


class QNet(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, obs_scaler: ObsScaler):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.obs_scaler = obs_scaler

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.obs_scaler(obs), act], dim=-1)
        feats = self.backbone(x)
        return self.head(feats).squeeze(-1)


class QNetPixel(nn.Module):
    def __init__(self, obs_encoder: nn.Module, head: nn.Module, obs_scaler: ObsScaler):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.head = head
        self.obs_scaler = obs_scaler

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        latent = self.obs_encoder(self.obs_scaler(obs))
        x = torch.cat([latent, act], dim=-1)
        return self.head(x).squeeze(-1)


@dataclasses.dataclass
class SACModules:
    actor_backbone: nn.Module
    actor_head: nn.Module
    q1_backbone: nn.Module
    q1_head: nn.Module
    q2_backbone: nn.Module
    q2_head: nn.Module
    obs_scaler: ObsScaler
    actor: ActorNet
    q1: nn.Module
    q2: nn.Module
    q1_target: nn.Module
    q2_target: nn.Module
    log_alpha: nn.Parameter


@dataclasses.dataclass
class SACOptimizers:
    actor_optimizer: optim.Optimizer
    critic_optimizer: optim.Optimizer
    alpha_optimizer: optim.Optimizer


@dataclasses.dataclass
class _QBundle:
    q1_backbone: nn.Module
    q1_head: nn.Module
    q2_backbone: nn.Module
    q2_head: nn.Module
    q1: nn.Module
    q2: nn.Module


def _resolve_backbone_name(config: SACConfig, obs_spec: Any) -> str:
    if getattr(obs_spec, "mode", "vector") != "pixels":
        return str(config.backbone_name)
    channels = int(getattr(obs_spec, "channels", 3) or 3)
    key = str(config.backbone_name).strip().lower()
    if key in {"mlp", "nature_cnn"} and channels == 4:
        return "nature_cnn_atari"
    if key in {"mlp", "nature_cnn_atari"} and channels != 4:
        return "nature_cnn"
    return str(config.backbone_name)


def _build_backbone_specs(config: SACConfig, obs_spec: Any) -> tuple[BackboneSpec, HeadSpec, HeadSpec]:
    backbone_spec = BackboneSpec(
        name=_resolve_backbone_name(config, obs_spec),
        hidden_sizes=tuple(config.backbone_hidden_sizes),
        activation=str(config.backbone_activation),
        layer_norm=bool(config.backbone_layer_norm),
    )
    actor_head_spec = HeadSpec(
        hidden_sizes=tuple(config.actor_head_hidden_sizes),
        activation=str(config.head_activation),
    )
    critic_head_spec = HeadSpec(
        hidden_sizes=tuple(config.critic_head_hidden_sizes),
        activation=str(config.head_activation),
    )
    return backbone_spec, actor_head_spec, critic_head_spec


def _build_q_modules(
    obs_spec: Any,
    env: Any,
    backbone_spec: BackboneSpec,
    critic_head_spec: HeadSpec,
    obs_scaler: ObsScaler,
) -> _QBundle:
    obs_dim = int(obs_spec.vector_dim or 64)
    if obs_spec.mode == "pixels":
        q1_backbone, q1_feat_dim = build_backbone(backbone_spec, input_dim=obs_dim)
        q2_backbone, q2_feat_dim = build_backbone(backbone_spec, input_dim=obs_dim)
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
    q1_backbone, q1_feat_dim = build_backbone(backbone_spec, input_dim=q_input_dim)
    q2_backbone, q2_feat_dim = build_backbone(backbone_spec, input_dim=q_input_dim)
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


def build_modules(config: SACConfig, env: Any, obs_spec: Any, *, device: torch.device) -> SACModules:
    obs_scaler = ObsScaler(env.obs_lb, env.obs_width).to(device)
    backbone_spec, actor_head_spec, critic_head_spec = _build_backbone_specs(config, obs_spec)

    obs_dim = int(obs_spec.vector_dim or 64)
    actor_backbone, actor_feat_dim = build_backbone(backbone_spec, input_dim=obs_dim)
    actor_head = build_mlp_head(actor_head_spec, input_dim=actor_feat_dim, output_dim=2 * int(env.act_dim))
    actor = ActorNet(actor_backbone, actor_head, obs_scaler, act_dim=int(env.act_dim))

    q_bundle = _build_q_modules(
        obs_spec,
        env,
        backbone_spec,
        critic_head_spec,
        obs_scaler,
    )
    q1_backbone = q_bundle.q1_backbone
    q1_head = q_bundle.q1_head
    q2_backbone = q_bundle.q2_backbone
    q2_head = q_bundle.q2_head
    q1 = q_bundle.q1
    q2 = q_bundle.q2

    actor.to(device)
    q1.to(device)
    q2.to(device)
    q1_target = copy.deepcopy(q1).to(device).eval()
    q2_target = copy.deepcopy(q2).to(device).eval()
    for p in q1_target.parameters():
        p.requires_grad_(False)
    for p in q2_target.parameters():
        p.requires_grad_(False)

    actor_param_count = sum(p.numel() for p in actor_backbone.parameters()) + sum(p.numel() for p in actor_head.parameters())
    if config.theta_dim is not None:
        assert int(actor_param_count) == int(config.theta_dim), (actor_param_count, config.theta_dim)

    log_alpha = nn.Parameter(torch.tensor(np.log(float(max(config.alpha_init, 1e-8))), dtype=torch.float32, device=device))
    return SACModules(
        actor_backbone=actor_backbone,
        actor_head=actor_head,
        q1_backbone=q1_backbone,
        q1_head=q1_head,
        q2_backbone=q2_backbone,
        q2_head=q2_head,
        obs_scaler=obs_scaler,
        actor=actor,
        q1=q1,
        q2=q2,
        q1_target=q1_target,
        q2_target=q2_target,
        log_alpha=log_alpha,
    )


def build_optimizers(config: SACConfig, modules: SACModules) -> SACOptimizers:
    actor_params = list(modules.actor_backbone.parameters()) + list(modules.actor_head.parameters())
    critic_params = list(modules.q1.parameters()) + list(modules.q2.parameters())
    return SACOptimizers(
        actor_optimizer=optim.AdamW(actor_params, lr=float(config.learning_rate_actor), weight_decay=0.0),
        critic_optimizer=optim.AdamW(critic_params, lr=float(config.learning_rate_critic), weight_decay=0.0),
        alpha_optimizer=optim.AdamW([modules.log_alpha], lr=float(config.learning_rate_alpha), weight_decay=0.0),
    )


def alpha(modules: SACModules) -> torch.Tensor:
    return modules.log_alpha.exp()


def sac_update(
    config: SACConfig,
    modules: SACModules,
    optimizers: SACOptimizers,
    replay,
    *,
    device: torch.device,
) -> tuple[float, float, float]:
    obs, act, rew, nxt, done = replay.sample(int(config.batch_size), device=device)
    target_entropy = float(config.target_entropy) if config.target_entropy is not None else -float(act.shape[-1])
    return sac_update_step(
        modules=SACUpdateModules(
            actor=modules.actor,
            q1=modules.q1,
            q2=modules.q2,
            q1_target=modules.q1_target,
            q2_target=modules.q2_target,
            log_alpha=modules.log_alpha,
        ),
        optimizers=SACUpdateOptimizers(
            actor=optimizers.actor_optimizer,
            critic=optimizers.critic_optimizer,
            alpha=optimizers.alpha_optimizer,
        ),
        batch=SACUpdateBatch(obs=obs, act=act, rew=rew, nxt=nxt, done=done),
        hyper=SACUpdateHyperParams(
            gamma=float(config.gamma),
            tau=float(config.tau),
            target_entropy=target_entropy,
        ),
    )


def capture_actor_state(modules: SACModules) -> dict[str, Any]:
    return {
        "backbone": {k: v.detach().clone() for k, v in modules.actor_backbone.state_dict().items()},
        "head": {k: v.detach().clone() for k, v in modules.actor_head.state_dict().items()},
        "obs_scaler": {k: v.detach().clone() for k, v in modules.obs_scaler.state_dict().items()},
    }


def restore_actor_state(modules: SACModules, snapshot: dict[str, Any]) -> None:
    modules.actor_backbone.load_state_dict(snapshot["backbone"])
    modules.actor_head.load_state_dict(snapshot["head"])
    if "obs_scaler" in snapshot:
        modules.obs_scaler.load_state_dict(snapshot["obs_scaler"])


@contextmanager
def use_actor_state(modules: SACModules, snapshot: dict[str, Any]):
    previous = capture_actor_state(modules)
    restore_actor_state(modules, snapshot)
    try:
        yield
    finally:
        restore_actor_state(modules, previous)
