from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.core.sac_update import SACUpdateBatch, SACUpdateHyperParams, SACUpdateModules, SACUpdateOptimizers, sac_update_step
from rl.pufferlib.offpolicy import model_utils as offpolicy_model_utils

from .config import SACConfig

ActorNet = offpolicy_model_utils.ActorNet
QNet = offpolicy_model_utils.QNet
QNetPixel = offpolicy_model_utils.QNetPixel
capture_actor_state = offpolicy_model_utils.capture_actor_state
restore_actor_state = offpolicy_model_utils.restore_actor_state
use_actor_state = offpolicy_model_utils.use_actor_state


@dataclasses.dataclass
class SACModules:
    actor_backbone: nn.Module
    actor_head: nn.Module
    q1_backbone: nn.Module
    q1_head: nn.Module
    q2_backbone: nn.Module
    q2_head: nn.Module
    obs_scaler: nn.Module
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


def build_modules(config: SACConfig, env: Any, obs_spec: Any, *, device: torch.device) -> SACModules:
    shared = offpolicy_model_utils.build_modules(config, env, obs_spec, device=device)
    log_alpha = nn.Parameter(torch.tensor(np.log(float(max(config.alpha_init, 1e-8))), dtype=torch.float32, device=device))
    return SACModules(
        actor_backbone=shared.actor_backbone,
        actor_head=shared.actor_head,
        q1_backbone=shared.q1_backbone,
        q1_head=shared.q1_head,
        q2_backbone=shared.q2_backbone,
        q2_head=shared.q2_head,
        obs_scaler=shared.obs_scaler,
        actor=shared.actor,
        q1=shared.q1,
        q2=shared.q2,
        q1_target=shared.q1_target,
        q2_target=shared.q2_target,
        log_alpha=log_alpha,
    )


def build_optimizers(config: SACConfig, modules: SACModules) -> SACOptimizers:
    shared = offpolicy_model_utils.build_optimizers(config, modules)
    return SACOptimizers(
        actor_optimizer=shared.actor_optimizer,
        critic_optimizer=shared.critic_optimizer,
        alpha_optimizer=optim.AdamW([modules.log_alpha], lr=float(config.learning_rate_alpha), weight_decay=0.0),
    )


def alpha(modules: SACModules) -> torch.Tensor:
    return modules.log_alpha.exp()


def sac_update(config: SACConfig, modules: SACModules, optimizers: SACOptimizers, replay, *, device: torch.device) -> tuple[float, float, float]:
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
        hyper=SACUpdateHyperParams(gamma=float(config.gamma), tau=float(config.tau), target_entropy=target_entropy),
    )
