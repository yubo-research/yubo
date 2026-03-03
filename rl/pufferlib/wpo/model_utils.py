from __future__ import annotations

import copy
import dataclasses
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from rl.core.wpo_update import WPOUpdateBatch, WPOUpdateHyperParams, WPOUpdateModules, WPOUpdateOptimizers, wpo_update_step
from rl.pufferlib.offpolicy import model_utils as offpolicy_model_utils

from .config import WPOConfig

ActorNet = offpolicy_model_utils.ActorNet
capture_actor_state = offpolicy_model_utils.capture_actor_state
restore_actor_state = offpolicy_model_utils.restore_actor_state
use_actor_state = offpolicy_model_utils.use_actor_state


@dataclasses.dataclass
class WPOModules:
    actor_backbone: nn.Module
    actor_head: nn.Module
    q1_backbone: nn.Module
    q1_head: nn.Module
    q2_backbone: nn.Module
    q2_head: nn.Module
    obs_scaler: nn.Module
    actor: ActorNet
    actor_target: ActorNet
    q1: nn.Module
    q2: nn.Module
    q1_target: nn.Module
    q2_target: nn.Module
    log_alpha_mean: nn.Parameter
    log_alpha_stddev: nn.Parameter


@dataclasses.dataclass
class WPOOptimizers:
    actor_optimizer: optim.Optimizer
    critic_optimizer: optim.Optimizer
    dual_optimizer: optim.Optimizer


def build_modules(config: WPOConfig, env: Any, obs_spec: Any, *, device: torch.device) -> WPOModules:
    shared_modules = offpolicy_model_utils.build_modules(config, env, obs_spec, device=device)
    actor_target = copy.deepcopy(shared_modules.actor).to(device).eval()
    for param in actor_target.parameters():
        param.requires_grad_(False)
    dual_shape = (int(env.act_dim),) if bool(config.per_dim_constraining) else (1,)
    log_alpha_mean = nn.Parameter(torch.full(dual_shape, float(config.init_log_alpha_mean), dtype=torch.float32, device=device))
    log_alpha_stddev = nn.Parameter(torch.full(dual_shape, float(config.init_log_alpha_stddev), dtype=torch.float32, device=device))
    return WPOModules(
        actor_backbone=shared_modules.actor_backbone,
        actor_head=shared_modules.actor_head,
        q1_backbone=shared_modules.q1_backbone,
        q1_head=shared_modules.q1_head,
        q2_backbone=shared_modules.q2_backbone,
        q2_head=shared_modules.q2_head,
        obs_scaler=shared_modules.obs_scaler,
        actor=shared_modules.actor,
        actor_target=actor_target,
        q1=shared_modules.q1,
        q2=shared_modules.q2,
        q1_target=shared_modules.q1_target,
        q2_target=shared_modules.q2_target,
        log_alpha_mean=log_alpha_mean,
        log_alpha_stddev=log_alpha_stddev,
    )


def build_optimizers(config: WPOConfig, modules: WPOModules) -> WPOOptimizers:
    actor_params = list(modules.actor_backbone.parameters()) + list(modules.actor_head.parameters())
    critic_params = list(modules.q1.parameters()) + list(modules.q2.parameters())
    dual_params = [modules.log_alpha_mean, modules.log_alpha_stddev]
    return WPOOptimizers(
        actor_optimizer=optim.AdamW(actor_params, lr=float(config.learning_rate_actor), weight_decay=0.0),
        critic_optimizer=optim.AdamW(critic_params, lr=float(config.learning_rate_critic), weight_decay=0.0),
        dual_optimizer=optim.AdamW(dual_params, lr=float(config.learning_rate_dual), weight_decay=0.0),
    )


def wpo_update(config: WPOConfig, modules: WPOModules, optimizers: WPOOptimizers, replay, *, device: torch.device) -> tuple[float, float, float, float, float]:
    obs, act, rew, nxt, done = replay.sample(int(config.batch_size), device=device)
    return wpo_update_step(
        modules=WPOUpdateModules(
            actor=modules.actor,
            actor_target=modules.actor_target,
            q1=modules.q1,
            q2=modules.q2,
            q1_target=modules.q1_target,
            q2_target=modules.q2_target,
            log_alpha_mean=modules.log_alpha_mean,
            log_alpha_stddev=modules.log_alpha_stddev,
        ),
        optimizers=WPOUpdateOptimizers(actor=optimizers.actor_optimizer, critic=optimizers.critic_optimizer, dual=optimizers.dual_optimizer),
        batch=WPOUpdateBatch(obs=obs, act=act, rew=rew, nxt=nxt, done=done),
        hyper=WPOUpdateHyperParams(
            gamma=float(config.gamma),
            tau=float(config.tau),
            num_samples=int(config.num_samples),
            epsilon_mean=float(config.epsilon_mean),
            epsilon_stddev=float(config.epsilon_stddev),
            policy_loss_scale=float(config.policy_loss_scale),
            kl_loss_scale=float(config.kl_loss_scale),
            dual_loss_scale=float(config.dual_loss_scale),
            per_dim_constraining=bool(config.per_dim_constraining),
            squashing_type=str(config.squashing_type),
        ),
    )
