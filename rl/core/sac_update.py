from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
import torch.optim as optim

from rl.core.sac_math import (
    compute_sac_target,
    sac_actor_loss,
    sac_alpha_loss,
    sac_critic_loss,
    soft_update_module,
)


def _set_requires_grad(modules: tuple[nn.Module, ...], enabled: bool) -> None:
    for module in modules:
        for p in module.parameters():
            p.requires_grad_(enabled)


@dataclasses.dataclass(frozen=True)
class SACUpdateModules:
    actor: nn.Module
    q1: nn.Module
    q2: nn.Module
    q1_target: nn.Module
    q2_target: nn.Module
    log_alpha: nn.Parameter


@dataclasses.dataclass(frozen=True)
class SACUpdateOptimizers:
    actor: optim.Optimizer
    critic: optim.Optimizer
    alpha: optim.Optimizer


@dataclasses.dataclass(frozen=True)
class SACUpdateBatch:
    obs: torch.Tensor
    act: torch.Tensor
    rew: torch.Tensor
    nxt: torch.Tensor
    done: torch.Tensor


@dataclasses.dataclass(frozen=True)
class SACUpdateHyperParams:
    gamma: float
    tau: float
    target_entropy: float


def sac_update_step(
    modules: SACUpdateModules,
    optimizers: SACUpdateOptimizers,
    batch: SACUpdateBatch,
    hyper: SACUpdateHyperParams,
) -> tuple[float, float, float]:
    with torch.no_grad():
        nxt_action, nxt_log_prob = modules.actor.sample(batch.nxt, deterministic=False)
        q1_t = modules.q1_target(batch.nxt, nxt_action)
        q2_t = modules.q2_target(batch.nxt, nxt_action)
        target = compute_sac_target(
            batch.rew,
            batch.done,
            gamma=float(hyper.gamma),
            q1_target=q1_t,
            q2_target=q2_t,
            alpha=modules.log_alpha.exp(),
            next_log_prob=nxt_log_prob,
        )
    q1_obs = modules.q1(batch.obs, batch.act)
    q2_obs = modules.q2(batch.obs, batch.act)
    critic_loss = sac_critic_loss(q1_obs, q2_obs, target)
    optimizers.critic.zero_grad(set_to_none=True)
    critic_loss.backward()
    optimizers.critic.step()
    _set_requires_grad((modules.q1, modules.q2), enabled=False)
    pi_action, log_prob = modules.actor.sample(batch.obs, deterministic=False)
    q_pi = torch.min(modules.q1(batch.obs, pi_action), modules.q2(batch.obs, pi_action))
    actor_loss = sac_actor_loss(modules.log_alpha.exp(), log_prob, q_pi)
    optimizers.actor.zero_grad(set_to_none=True)
    actor_loss.backward()
    optimizers.actor.step()
    _set_requires_grad((modules.q1, modules.q2), enabled=True)
    alpha_loss = sac_alpha_loss(modules.log_alpha, log_prob, target_entropy=float(hyper.target_entropy))
    optimizers.alpha.zero_grad(set_to_none=True)
    alpha_loss.backward()
    optimizers.alpha.step()
    soft_update_module(modules.q1_target, modules.q1, tau=float(hyper.tau))
    soft_update_module(modules.q2_target, modules.q2, tau=float(hyper.tau))
    return (
        float(actor_loss.item()),
        float(critic_loss.item()),
        float(alpha_loss.item()),
    )
