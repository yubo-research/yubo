from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import PufferPPOConfig
from .specs import _FlatBatch, _RolloutBuffer, _RuntimeState, _TrainPlan, _UpdateStats


def _update_episode_stats(state: _RuntimeState, infos: list | None) -> None:
    for item in infos or []:
        if isinstance(item, dict) and "episode_return" in item:
            state.last_episode_return = float(item["episode_return"])


def collect_rollout(plan: _TrainPlan, model, envs, buffer: _RolloutBuffer, state: _RuntimeState, device: torch.device, *, prepare_obs_fn) -> None:
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
        buffer.rewards[step] = torch.as_tensor(reward_np, dtype=torch.float32, device=device).view(-1)
        done_np = np.logical_or(term_np, trunc_np)
        state.next_done = torch.as_tensor(done_np, dtype=torch.float32, device=device).view(-1)
        state.next_obs = prepare_obs_fn(next_obs_np, obs_spec=state.obs_spec, device=device)
        _update_episode_stats(state, infos)


def compute_advantages(
    plan: _TrainPlan, config: PufferPPOConfig, model, state: _RuntimeState, buffer: _RolloutBuffer, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        next_value = model.get_value(state.next_obs)
        advantages = torch.zeros_like(buffer.rewards)
        lastgaelam = torch.zeros(plan.num_envs, dtype=torch.float32, device=device)
        for t in reversed(range(plan.num_steps)):
            if t == plan.num_steps - 1:
                next_nonterminal = 1.0 - state.next_done
                next_values = next_value
            else:
                next_nonterminal = 1.0 - buffer.dones[t + 1]
                next_values = buffer.values[t + 1]
            delta = buffer.rewards[t] + float(config.gamma) * next_values * next_nonterminal - buffer.values[t]
            lastgaelam = delta + float(config.gamma) * float(config.gae_lambda) * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + buffer.values
    return (advantages, returns)


def flatten_batch(plan: _TrainPlan, buffer: _RolloutBuffer, advantages: torch.Tensor, returns: torch.Tensor, obs_shape: tuple[int, ...]) -> _FlatBatch:
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


def _clipped_value_loss(config: PufferPPOConfig, batch: _FlatBatch, mb_inds: np.ndarray, newvalue: torch.Tensor) -> torch.Tensor:
    v_loss_unclipped = (newvalue - batch.returns[mb_inds]) ** 2
    v_clipped = batch.values[mb_inds] + torch.clamp(newvalue - batch.values[mb_inds], -float(config.clip_coef), float(config.clip_coef))
    v_loss_clipped = (v_clipped - batch.returns[mb_inds]) ** 2
    return 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()


def _compute_losses(
    config: PufferPPOConfig, batch: _FlatBatch, mb_inds: np.ndarray, ratio: torch.Tensor, entropy: torch.Tensor, newvalue: torch.Tensor
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


def _track_kl_clip(logratio: torch.Tensor, ratio: torch.Tensor, config: PufferPPOConfig, clipfrac_values: list[float]) -> float:
    with torch.no_grad():
        approx_kl = (ratio - 1 - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > float(config.clip_coef)).float().mean().item()
        clipfrac_values.append(float(clipfrac))
        return float(approx_kl.item())


def ppo_update(config: PufferPPOConfig, plan: _TrainPlan, model, optimizer: optim.Optimizer, batch: _FlatBatch, b_inds: np.ndarray) -> _UpdateStats:
    approx_kl_value = 0.0
    clipfrac_values: list[float] = []
    for _epoch in range(int(config.update_epochs)):
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
