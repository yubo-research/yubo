from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .specs import _FlatBatch, _RolloutBuffer, _RuntimeState, _TrainPlan, _UpdateStats

_PUFFER_ADV_AVAILABLE: bool | None = None


def _gae_python_fallback(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_done: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """Python loop GAE when puffer kernel unavailable. Compilable for potential fusion."""
    n_steps, n_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(n_envs, dtype=torch.float32, device=rewards.device)
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_nonterminal = 1.0 - next_done
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
        advantages[t] = lastgaelam
    return advantages


def _puffer_advantage_available() -> bool:
    """True if pufferlib C extension and compute_puff_advantage are available."""
    global _PUFFER_ADV_AVAILABLE
    if _PUFFER_ADV_AVAILABLE is not None:
        return _PUFFER_ADV_AVAILABLE
    try:
        from pufferlib import _C  # noqa: F401

        _ = torch.ops.pufferlib.compute_puff_advantage
        _PUFFER_ADV_AVAILABLE = True
    except Exception:
        _PUFFER_ADV_AVAILABLE = False
    return _PUFFER_ADV_AVAILABLE


def _numpy_to_device(arr: np.ndarray, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Move numpy to device; use pin_memory for CPU→CUDA when PPO_PIN_MEMORY=1 (default off; benchmark showed no gain for cheetah)."""
    t = torch.as_tensor(arr, dtype=dtype)
    if device.type == "cuda" and os.environ.get("PPO_PIN_MEMORY", "0") in ("1", "true", "yes"):
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def _update_episode_stats(state: _RuntimeState, infos: list | None) -> None:
    for item in infos or []:
        if isinstance(item, dict) and "episode_return" in item:
            state.last_episode_return = float(item["episode_return"])


def collect_rollout(
    plan: _TrainPlan,
    model,
    envs,
    buffer: _RolloutBuffer,
    state: _RuntimeState,
    device: torch.device,
    *,
    prepare_obs_fn,
) -> None:
    use_batched_cpu = os.environ.get("PPO_BATCHED_CPU_BUFFER", "0") in ("1", "true", "yes")
    rewards_np = np.zeros((plan.num_steps, plan.num_envs), dtype=np.float32) if use_batched_cpu else None
    dones_np = np.zeros((plan.num_steps, plan.num_envs), dtype=np.float32) if use_batched_cpu else None

    for step in range(plan.num_steps):
        state.global_step += plan.num_envs
        buffer.obs[step] = state.next_obs
        buffer.dones[step] = state.next_done
        with torch.no_grad():
            action, logprob, _, value = model.get_action_and_value(state.next_obs)
        buffer.actions[step] = action
        buffer.logprobs[step] = logprob
        buffer.values[step] = value
        next_obs_np, reward_np, term_np, trunc_np, infos = envs.step(action.detach().cpu().numpy())
        done_np = np.logical_or(term_np, trunc_np)
        if use_batched_cpu:
            rewards_np[step] = np.asarray(reward_np, dtype=np.float32).reshape(plan.num_envs)
            dones_np[step] = np.asarray(done_np, dtype=np.float32).reshape(plan.num_envs)
        else:
            buffer.rewards[step] = _numpy_to_device(reward_np, device=device).view(-1)
            state.next_done = _numpy_to_device(done_np, device=device).view(-1)
        state.next_obs = prepare_obs_fn(next_obs_np, obs_spec=state.obs_spec, device=device)
        _update_episode_stats(state, infos)

    if use_batched_cpu:
        buffer.rewards.copy_(torch.as_tensor(rewards_np, device=device))
        # buffer.dones[t] = done after step t-1; dones_np[t] = done after step t
        buffer.dones[1:].copy_(torch.as_tensor(dones_np[:-1], device=device))
        state.next_done = torch.as_tensor(dones_np[-1], device=device).view(-1)


def collect_rollout_async(
    plan: _TrainPlan,
    model,
    envs,
    buffer: _RolloutBuffer,
    state: _RuntimeState,
    device: torch.device,
    *,
    prepare_obs_fn,
) -> None:
    """Async rollout using pufferlib send/recv. Overlaps env stepping with model forward."""
    # recv returns (obs, rewards, terms, truncs, infos, env_ids, masks)
    for step in range(plan.num_steps):
        state.global_step += plan.num_envs
        buffer.obs[step] = state.next_obs
        buffer.dones[step] = state.next_done
        with torch.no_grad():
            action, logprob, _, value = model.get_action_and_value(state.next_obs)
        buffer.actions[step] = action
        buffer.logprobs[step] = logprob
        buffer.values[step] = value
        envs.send(action.detach().cpu().numpy())
        out = envs.recv()
        next_obs_np = np.asarray(out[0])
        reward_np = np.asarray(out[1], dtype=np.float32)
        term_np = np.asarray(out[2])
        trunc_np = np.asarray(out[3])
        infos = out[4] if len(out) > 4 else []
        buffer.rewards[step] = _numpy_to_device(reward_np, device=device).view(-1)
        done_np = np.logical_or(term_np, trunc_np)
        state.next_done = _numpy_to_device(done_np, device=device).view(-1)
        state.next_obs = prepare_obs_fn(next_obs_np, obs_spec=state.obs_spec, device=device)
        _update_episode_stats(state, infos)


def compute_advantages(
    plan: _TrainPlan,
    config: Any,
    model,
    state: _RuntimeState,
    buffer: _RolloutBuffer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    use_puffer = _puffer_advantage_available() and os.environ.get("PPO_USE_PUFFER_ADVANTAGE", "1") not in ("0", "false", "no")
    gamma = float(config.gamma)
    gae_lambda = float(config.gae_lambda)

    with torch.no_grad():
        next_value = model.get_value(state.next_obs)
        advantages = torch.zeros_like(buffer.rewards)

        if use_puffer:
            # Pufferlib kernel: [num_segments, horizon]. Each row = one env's trace.
            # We need horizon = num_steps+1 for bootstrap; kernel writes adv[0..horizon-2].
            n_envs, n_steps = plan.num_envs, plan.num_steps
            horizon = n_steps + 1
            # values: [n_envs, horizon], last col = next_value
            values_puff = torch.empty((n_envs, horizon), dtype=torch.float32, device=device)
            values_puff[:, :n_steps] = buffer.values.T
            values_puff[:, n_steps] = next_value
            # rewards: kernel uses rewards[t_next] for delta at t; rewards[1]=r(0->1), etc.
            rewards_puff = torch.empty((n_envs, horizon), dtype=torch.float32, device=device)
            rewards_puff[:, 0] = 0.0
            rewards_puff[:, 1 : n_steps + 1] = buffer.rewards.T
            # dones: [n_envs, horizon], dones[:,t+1] = done after step t
            dones_puff = torch.empty((n_envs, horizon), dtype=torch.float32, device=device)
            dones_puff[:, 0] = 0.0
            dones_puff[:, 1:n_steps] = buffer.dones[1:n_steps].T
            dones_puff[:, n_steps] = state.next_done
            # importance = 1 for standard GAE (no vtrace clipping)
            importance_puff = torch.ones((n_envs, horizon), dtype=torch.float32, device=device)
            advantages_puff = torch.zeros((n_envs, horizon), dtype=torch.float32, device=device)
            torch.ops.pufferlib.compute_puff_advantage(
                values_puff,
                rewards_puff,
                dones_puff,
                importance_puff,
                advantages_puff,
                gamma,
                gae_lambda,
                1.0,
                1.0,
            )
            advantages[:] = advantages_puff[:, :n_steps].T
        else:
            advantages[:] = _gae_python_fallback(
                buffer.rewards,
                buffer.values,
                buffer.dones,
                state.next_done,
                next_value,
                gamma,
                gae_lambda,
            )

        returns = advantages + buffer.values
    return (advantages, returns)


def flatten_batch(
    plan: _TrainPlan,
    buffer: _RolloutBuffer,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    obs_shape: tuple[int, ...],
) -> _FlatBatch:
    if buffer.actions.ndim == 2:
        flat_actions = buffer.actions.reshape(-1)
    else:
        flat_actions = buffer.actions.reshape(plan.batch_size, *buffer.actions.shape[2:])
    return _FlatBatch(
        obs=buffer.obs.reshape((plan.batch_size, *obs_shape)),
        actions=flat_actions,
        logprobs=buffer.logprobs.reshape(-1),
        advantages=advantages.reshape(-1),
        returns=returns.reshape(-1),
        values=buffer.values.reshape(-1),
    )


def _clipped_value_loss(config: Any, batch: _FlatBatch, mb_inds: np.ndarray, newvalue: torch.Tensor) -> torch.Tensor:
    v_loss_unclipped = (newvalue - batch.returns[mb_inds]) ** 2
    v_clipped = batch.values[mb_inds] + torch.clamp(
        newvalue - batch.values[mb_inds],
        -float(config.clip_coef),
        float(config.clip_coef),
    )
    v_loss_clipped = (v_clipped - batch.returns[mb_inds]) ** 2
    return 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()


def _compute_losses(
    config: Any,
    batch: _FlatBatch,
    mb_inds: np.ndarray,
    ratio: torch.Tensor,
    entropy: torch.Tensor,
    newvalue: torch.Tensor,
) -> torch.Tensor:
    mb_advantages = batch.advantages[mb_inds]
    if bool(config.norm_adv):
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-08)
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - float(config.clip_coef), 1.0 + float(config.clip_coef))
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    newvalue = newvalue.view(-1)
    if bool(config.clip_vloss):
        v_loss = _clipped_value_loss(config, batch, mb_inds, newvalue)
    else:
        v_loss = 0.5 * ((newvalue - batch.returns[mb_inds]) ** 2).mean()
    entropy_loss = entropy.mean()
    return pg_loss - float(config.ent_coef) * entropy_loss + float(config.vf_coef) * v_loss


def _track_kl_clip(
    logratio: torch.Tensor,
    ratio: torch.Tensor,
    config: Any,
    clipfrac_values: list[float],
) -> float:
    with torch.no_grad():
        approx_kl = (ratio - 1 - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > float(config.clip_coef)).float().mean().item()
        clipfrac_values.append(float(clipfrac))
        return float(approx_kl.item())


def ppo_update(
    config: Any,
    plan: _TrainPlan,
    model,
    optimizer: optim.Optimizer,
    batch: _FlatBatch,
    b_inds: np.ndarray,
) -> _UpdateStats:
    approx_kl_value = 0.0
    clipfrac_values: list[float] = []
    for _epoch in range(int(config.epochs)):
        np.random.shuffle(b_inds)
        for start in range(0, plan.batch_size, plan.minibatch_size):
            mb_inds = b_inds[start : start + plan.minibatch_size]
            _, newlogprob, entropy, newvalue = model.get_action_and_value(batch.obs[mb_inds], action=batch.actions[mb_inds])
            logratio = newlogprob - batch.logprobs[mb_inds]
            ratio = logratio.exp()
            approx_kl_value = _track_kl_clip(logratio, ratio, config, clipfrac_values)
            loss = _compute_losses(config, batch, mb_inds, ratio, entropy, newvalue)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
            optimizer.step()
        if config.target_kl is not None and approx_kl_value > float(config.target_kl):
            break
    clipfrac_mean = float(np.mean(clipfrac_values)) if clipfrac_values else 0.0
    return _UpdateStats(approx_kl=float(approx_kl_value), clipfrac_mean=float(clipfrac_mean))
