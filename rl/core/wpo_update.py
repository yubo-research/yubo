from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from rl.core.offpolicy_objectives import polyak_update_parameters

_WPO_FLOAT_EPSILON = 1e-8


def _set_requires_grad(modules: tuple[nn.Module, ...], enabled: bool) -> None:
    for module in modules:
        for p in module.parameters():
            p.requires_grad_(enabled)


@dataclasses.dataclass(frozen=True)
class WPOUpdateModules:
    actor: nn.Module
    actor_target: nn.Module
    q1: nn.Module
    q2: nn.Module
    q1_target: nn.Module
    q2_target: nn.Module
    log_alpha_mean: nn.Parameter
    log_alpha_stddev: nn.Parameter


@dataclasses.dataclass(frozen=True)
class WPOUpdateOptimizers:
    actor: optim.Optimizer
    critic: optim.Optimizer
    dual: optim.Optimizer


@dataclasses.dataclass(frozen=True)
class WPOUpdateBatch:
    obs: torch.Tensor
    act: torch.Tensor
    rew: torch.Tensor
    nxt: torch.Tensor
    done: torch.Tensor


@dataclasses.dataclass(frozen=True)
class WPOUpdateHyperParams:
    gamma: float
    tau: float
    num_samples: int
    epsilon_mean: float
    epsilon_stddev: float
    policy_loss_scale: float
    kl_loss_scale: float
    dual_loss_scale: float
    per_dim_constraining: bool
    squashing_type: str


def _squash_gradient(q_grad: torch.Tensor, squashing_type: str) -> torch.Tensor:
    key = str(squashing_type).strip().lower()
    if key in {"identity", "none"}:
        return q_grad
    if key in {"cbrt", "cube_root"}:
        return torch.sign(q_grad) * torch.abs(q_grad).pow(1.0 / 3.0)
    raise ValueError(f"Unsupported WPO squashing_type: {squashing_type}")


def _critic_target(modules: WPOUpdateModules, batch: WPOUpdateBatch, *, gamma: float) -> torch.Tensor:
    with torch.no_grad():
        nxt_action, _ = modules.actor_target.sample(batch.nxt, deterministic=False)
        q1_t = modules.q1_target(batch.nxt, nxt_action)
        q2_t = modules.q2_target(batch.nxt, nxt_action)
        q_t = torch.min(q1_t, q2_t)
        return batch.rew + (1.0 - batch.done) * float(gamma) * q_t


def _repeat_observations(obs: torch.Tensor, num_samples: int) -> torch.Tensor:
    batch_size = int(obs.shape[0])
    return obs.unsqueeze(0).expand(num_samples, *obs.shape).reshape(num_samples * batch_size, *obs.shape[1:])


def _policy_loss_from_action_jacobians(
    modules: WPOUpdateModules,
    *,
    obs_rep: torch.Tensor,
    done: torch.Tensor,
    num_samples: int,
    squashing_type: str,
) -> torch.Tensor:
    batch_size = int(done.shape[0])
    sampled_actions, _ = modules.actor.sample(obs_rep, deterministic=False)

    action_for_q_grad = sampled_actions.detach().requires_grad_(True)
    q_values = torch.min(modules.q1(obs_rep.detach(), action_for_q_grad), modules.q2(obs_rep.detach(), action_for_q_grad))
    q_values_grad = torch.autograd.grad(q_values.sum(), action_for_q_grad, create_graph=False, retain_graph=False)[0].detach()

    action_for_log_prob_grad = sampled_actions.detach().requires_grad_(True)
    log_probs = modules.actor.log_prob_from_action(obs_rep, action_for_log_prob_grad)
    log_prob_action_grads = torch.autograd.grad(log_probs.sum(), action_for_log_prob_grad, create_graph=True, retain_graph=True)[0]

    policy_term = -(log_prob_action_grads * _squash_gradient(q_values_grad, squashing_type)).sum(dim=-1).view(num_samples, batch_size)
    not_terminal = (1.0 - done).view(1, batch_size)
    return (policy_term.sum(dim=0) * not_terminal.view(-1)).sum() / not_terminal.sum().clamp_min(1.0)


def _mean_kl_terms(
    modules: WPOUpdateModules,
    *,
    obs: torch.Tensor,
    per_dim_constraining: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    online_mean, online_log_std = modules.actor.mean_log_std(obs)
    target_mean, target_log_std = modules.actor_target.mean_log_std(obs)
    online_std = torch.exp(online_log_std)
    target_std = torch.exp(target_log_std)

    target_dist = Normal(target_mean, target_std)
    fixed_stddev_dist = Normal(online_mean, target_std)
    fixed_mean_dist = Normal(target_mean, online_std)

    kl_mean = kl_divergence(target_dist, fixed_stddev_dist)
    kl_stddev = kl_divergence(target_dist, fixed_mean_dist)
    if not per_dim_constraining:
        kl_mean = kl_mean.sum(dim=-1, keepdim=True)
        kl_stddev = kl_stddev.sum(dim=-1, keepdim=True)
    return (kl_mean.mean(dim=0), kl_stddev.mean(dim=0))


def _policy_and_dual_losses(
    modules: WPOUpdateModules, batch: WPOUpdateBatch, hyper: WPOUpdateHyperParams
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = batch.obs
    done = batch.done
    num_samples = max(1, int(hyper.num_samples))
    obs_rep = _repeat_observations(obs, num_samples)
    loss_policy = _policy_loss_from_action_jacobians(
        modules,
        obs_rep=obs_rep,
        done=done,
        num_samples=num_samples,
        squashing_type=hyper.squashing_type,
    )
    mean_kl_mean, mean_kl_stddev = _mean_kl_terms(modules, obs=obs, per_dim_constraining=bool(hyper.per_dim_constraining))

    alpha_mean = torch.nn.functional.softplus(modules.log_alpha_mean) + _WPO_FLOAT_EPSILON
    alpha_stddev = torch.nn.functional.softplus(modules.log_alpha_stddev) + _WPO_FLOAT_EPSILON

    loss_kl = (alpha_mean.detach() * mean_kl_mean).sum() + (alpha_stddev.detach() * mean_kl_stddev).sum()
    loss_dual = (alpha_mean * (float(hyper.epsilon_mean) - mean_kl_mean.detach())).sum() + (
        alpha_stddev * (float(hyper.epsilon_stddev) - mean_kl_stddev.detach())
    ).sum()

    loss_actor = float(hyper.policy_loss_scale) * loss_policy + float(hyper.kl_loss_scale) * loss_kl
    return (loss_actor, loss_policy.detach(), loss_dual, alpha_mean.detach().mean(), alpha_stddev.detach().mean())


def wpo_update_step(
    modules: WPOUpdateModules,
    optimizers: WPOUpdateOptimizers,
    batch: WPOUpdateBatch,
    hyper: WPOUpdateHyperParams,
) -> tuple[float, float, float, float, float]:
    target = _critic_target(modules, batch, gamma=float(hyper.gamma))

    q1_obs = modules.q1(batch.obs, batch.act)
    q2_obs = modules.q2(batch.obs, batch.act)
    critic_loss = 0.5 * ((target - q1_obs).pow(2).mean() + (target - q2_obs).pow(2).mean())

    optimizers.critic.zero_grad(set_to_none=True)
    critic_loss.backward()
    optimizers.critic.step()

    _set_requires_grad((modules.q1, modules.q2), enabled=False)
    actor_loss, _policy_loss, dual_loss, alpha_mean, alpha_stddev = _policy_and_dual_losses(modules, batch, hyper)
    optimizers.actor.zero_grad(set_to_none=True)
    actor_loss.backward()
    optimizers.actor.step()
    _set_requires_grad((modules.q1, modules.q2), enabled=True)

    optimizers.dual.zero_grad(set_to_none=True)
    dual_obj = float(hyper.dual_loss_scale) * dual_loss
    dual_obj.backward()
    optimizers.dual.step()

    with torch.no_grad():
        modules.log_alpha_mean.data.clamp_(min=-18.0)
        modules.log_alpha_stddev.data.clamp_(min=-18.0)

    polyak_update_parameters(modules.q1_target, modules.q1, tau=float(hyper.tau))
    polyak_update_parameters(modules.q2_target, modules.q2, tau=float(hyper.tau))
    polyak_update_parameters(modules.actor_target, modules.actor, tau=float(hyper.tau))

    return (
        float(actor_loss.item()),
        float(critic_loss.item()),
        float(dual_loss.item()),
        float(alpha_mean.item()),
        float(alpha_stddev.item()),
    )
