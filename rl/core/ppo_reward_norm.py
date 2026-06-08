from __future__ import annotations

import torch


def normalize_rewards_for_training(config, state, batch, *, device: torch.device) -> None:
    if not bool(config.env.normalize_reward):
        return
    try:
        rewards = batch[("next", "reward")]
    except Exception:
        return
    try:
        dones = batch[("next", "done")].to(dtype=rewards.dtype, device=device)
    except Exception:
        dones = torch.zeros_like(rewards)
    returns = _discounted_reward_returns(config, state, rewards, dones, device=device)
    batch_var = torch.var(returns.reshape(-1), unbiased=False)
    batch_count = int(returns.numel())
    old_var = state.reward_var
    if old_var is None:
        old_var = torch.ones((), dtype=rewards.dtype, device=device)
    old_count = float(state.reward_count)
    total_count = old_count + float(batch_count)
    new_var = (old_var * old_count + batch_var * float(batch_count)) / total_count
    state.reward_var = new_var.detach()
    state.reward_count = total_count
    batch[("next", "reward")] = rewards / torch.sqrt(new_var + 1e-8)


def _discounted_reward_returns(
    config,
    state,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    reward_return = state.reward_return
    if reward_return is None or tuple(reward_return.shape) != tuple(rewards[0].shape):
        reward_return = torch.zeros_like(rewards[0], device=device)
    else:
        reward_return = reward_return.to(device=device, dtype=rewards.dtype)
    values = []
    gamma = float(config.loss.gamma)
    for reward_t, done_t in zip(rewards, dones, strict=True):
        reward_return = reward_t + gamma * reward_return
        values.append(reward_return)
        reward_return = reward_return * (1.0 - done_t)
    state.reward_return = reward_return.detach()
    return torch.stack(values, dim=0)
