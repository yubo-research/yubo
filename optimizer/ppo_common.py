from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from optimizer.designer_errors import NoSuchDesignerError
from optimizer.trajectory import Trajectory


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
    last_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    num_steps = rewards.shape[0]
    advantages = np.zeros_like(rewards, dtype=np.float32)
    lastgaelam = 0.0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_nonterminal = 1.0 - float(dones[t])
            next_values = last_value
        else:
            next_nonterminal = 1.0 - float(dones[t])
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
        advantages[t] = lastgaelam

    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def _discounted_segment_return(rewards: np.ndarray, gamma: float) -> float:
    if gamma == 1.0:
        return float(np.sum(rewards))
    ep_return = 0.0
    for r in reversed(rewards):
        ep_return = float(r) + gamma * ep_return
    return ep_return


def compute_episode_return_advantages(
    rewards: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float = 1.0,
) -> np.ndarray:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    start = 0
    for i in range(rewards.shape[0]):
        if dones[i]:
            segment = rewards[start : i + 1]
            ep_return = _discounted_segment_return(segment, gamma)
            advantages[start : i + 1] = ep_return
            start = i + 1
    if start < rewards.shape[0]:
        ep_return = _discounted_segment_return(rewards[start:], gamma)
        advantages[start:] = ep_return
    return advantages


def merge_trajectories(trajectories: list[Trajectory]) -> Trajectory:
    if not trajectories:
        raise NoSuchDesignerError("merge_trajectories requires at least one trajectory")
    if len(trajectories) == 1:
        return trajectories[0]

    all_states = [traj.states for traj in trajectories]
    all_actions = [traj.actions for traj in trajectories]
    all_rewards = [traj.rewards for traj in trajectories]
    all_log_probs = [traj.log_probs for traj in trajectories]
    all_values = [traj.values for traj in trajectories]

    has_values = [v is not None for v in all_values]
    if any(has_values) and not all(has_values):
        raise NoSuchDesignerError("Cannot merge trajectories with mixed value/no-value rollouts")

    merged_states = np.concatenate(all_states, axis=1)
    merged_actions = np.concatenate(all_actions, axis=1)
    merged_rewards = np.concatenate(all_rewards, axis=0)
    merged_log_probs = np.concatenate(all_log_probs, axis=0)
    dones_parts = []
    for traj in trajectories:
        d = np.asarray(traj.dones, dtype=bool).copy()
        d[-1] = True
        dones_parts.append(d)
    merged_dones = np.concatenate(dones_parts)
    if all(has_values):
        merged_values = np.concatenate(all_values, axis=0)
    else:
        merged_values = None

    total_return = sum(float(traj.rreturn) for traj in trajectories)
    total_steps = sum(traj.num_steps for traj in trajectories)

    return Trajectory(
        rreturn=total_return,
        states=merged_states,
        actions=merged_actions,
        num_steps=total_steps,
        rewards=merged_rewards,
        log_probs=merged_log_probs,
        values=merged_values,
        dones=merged_dones,
    )


def normalize_advantages(advantages: np.ndarray, device: torch.device) -> torch.Tensor:
    adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    if adv_t.numel() == 1:
        return torch.zeros_like(adv_t)
    std = adv_t.std(unbiased=False)
    if std < 1e-8:
        return adv_t
    return (adv_t - adv_t.mean()) / (std + 1e-8)


def resolve_designer_config(config, config_cls, overrides):
    if config is not None:
        return config
    return config_cls(**overrides)


def apply_ppo_telemetry(telemetry, dt_rollout: float, dt_update: float, num_arms: int) -> None:
    telemetry.set_dt_rollout(dt_rollout)
    telemetry.set_dt_fit(dt_update)
    telemetry.set_dt_select(0.0)
    telemetry.set_num_rollout_workers(num_arms)


def trajectory_tensors(merged_traj: Trajectory, device: torch.device) -> tuple:
    return (
        torch.as_tensor(merged_traj.states.T, dtype=torch.float32, device=device),
        torch.as_tensor(merged_traj.actions.T, dtype=torch.float32, device=device),
        torch.as_tensor(merged_traj.log_probs, dtype=torch.float32, device=device),
    )


@dataclass
class _PPOACBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


@dataclass
class _PPOPGBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor


def _ppo_ac_update_epoch(policy, optimizer, batch: _PPOACBatch, cfg) -> None:
    perm = torch.randperm(batch.obs.shape[0])
    mb_obs, mb_actions = batch.obs[perm], batch.actions[perm]
    mb_old_log_probs, mb_advantages = batch.old_log_probs[perm], batch.advantages[perm]
    mb_returns = batch.returns[perm]

    _, new_log_probs, entropy, new_values = policy.get_action_and_value(mb_obs, action=mb_actions)

    ratio = (new_log_probs - mb_old_log_probs).exp()
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
    loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy.mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
    optimizer.step()


def _ppo_pg_update_epoch(policy, optimizer, batch: _PPOPGBatch, cfg) -> None:
    perm = torch.randperm(batch.obs.shape[0])
    mb_obs, mb_actions = batch.obs[perm], batch.actions[perm]
    mb_old_log_probs, mb_advantages = batch.old_log_probs[perm], batch.advantages[perm]

    _, new_log_probs, entropy = policy.get_action_and_value(mb_obs, action=mb_actions)

    ratio = (new_log_probs - mb_old_log_probs).exp()
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    loss = pg_loss - cfg.ent_coef * entropy.mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
    optimizer.step()
