from __future__ import annotations

from typing import Any

import torch


def _get_nested(batch: Any, key: tuple[str, ...]):
    try:
        return batch[key]
    except Exception:
        return None


def _time_env_matrix(value: Any, *, num_envs: int, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=dtype)
    if tensor.ndim > 0 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.reshape(-1, 1)
    if int(num_envs) > 1 and tensor.shape[0] == int(num_envs) and tensor.shape[-1] != int(num_envs):
        tensor = tensor.transpose(0, 1)
    return tensor.reshape(-1, tensor.shape[-1])


def update_onpolicy_rollout_metrics(state: Any, batch: Any, *, num_envs: int) -> dict[str, float | None]:
    reward = _get_nested(batch, ("next", "reward"))
    done = _get_nested(batch, ("next", "done"))
    if reward is None or done is None:
        return {
            "rollout_reward": None,
            "nonfinite_reward_fraction": None,
            "rollout_return": None,
            "rollout_length": None,
        }

    rewards = _time_env_matrix(reward, num_envs=int(num_envs), dtype=torch.float32)
    dones = _time_env_matrix(done, num_envs=int(num_envs), dtype=torch.bool)
    if rewards.numel() == 0:
        return {
            "rollout_reward": None,
            "nonfinite_reward_fraction": None,
            "rollout_return": None,
            "rollout_length": None,
        }
    if dones.shape != rewards.shape:
        dones = dones.reshape(rewards.shape)
    finite_rewards = torch.isfinite(rewards)
    safe_rewards = torch.where(finite_rewards, rewards, torch.zeros_like(rewards))
    valid_rewards = rewards[finite_rewards]

    env_count = int(rewards.shape[1])
    returns = getattr(state, "rollout_returns", None)
    lengths = getattr(state, "rollout_lengths", None)
    if returns is None or int(returns.numel()) != env_count:
        returns = torch.zeros(env_count, dtype=torch.float32)
        lengths = torch.zeros(env_count, dtype=torch.float32)

    completed_returns: list[torch.Tensor] = []
    completed_lengths: list[torch.Tensor] = []
    for reward_t, done_t in zip(safe_rewards, dones, strict=True):
        returns += reward_t
        lengths += 1.0
        if bool(done_t.any()):
            completed_returns.append(returns[done_t].clone())
            completed_lengths.append(lengths[done_t].clone())
            returns[done_t] = 0.0
            lengths[done_t] = 0.0

    state.rollout_returns = returns
    state.rollout_lengths = lengths
    state.last_rollout_reward = float(valid_rewards.mean()) if int(valid_rewards.numel()) else None
    state.last_nonfinite_reward_fraction = float((~finite_rewards).float().mean())
    if completed_returns:
        state.last_rollout_return = float(torch.cat(completed_returns).mean())
        state.last_rollout_length = float(torch.cat(completed_lengths).mean())
    return {
        "rollout_reward": state.last_rollout_reward,
        "nonfinite_reward_fraction": state.last_nonfinite_reward_fraction,
        "rollout_return": getattr(state, "last_rollout_return", None),
        "rollout_length": getattr(state, "last_rollout_length", None),
    }
