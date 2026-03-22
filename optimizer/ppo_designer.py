import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from optimizer.designer_errors import NoSuchDesignerError
from optimizer.designer_protocol import Designer
from optimizer.trajectories import collect_trajectory


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip_coef: float = 0.2
    epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5


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


@dataclass
class _PPOBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


def _ppo_update_epoch(policy, optimizer, batch: _PPOBatch, cfg: PPOConfig) -> None:
    # Full-batch update (no minibatching): In the BO designer context, rollouts are short
    # (single episode per iteration), so minibatching adds complexity without benefit.
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


class PPODesigner(Designer):
    def __init__(
        self,
        policy,
        env_conf,
        *,
        config: PPOConfig | None = None,
        **overrides,
    ):
        self._policy = policy
        self._env_conf = env_conf
        if config is not None:
            self._config = config
        else:
            self._config = PPOConfig(**overrides)

    def _validate_policy(self, policy) -> None:
        required = ["get_action_and_value", "last_log_probs", "last_values"]
        missing = [m for m in required if not (hasattr(policy, m) and callable(getattr(policy, m)))]
        if missing:
            raise NoSuchDesignerError(
                f"PPO requires a policy with {', '.join(required)} methods "
                f"(missing: {', '.join(missing)}). Use ActorCriticMLPPolicy (e.g., env_tag='lunar-ac')."
            )

    def __call__(self, data, num_arms, *, telemetry=None):
        if num_arms != 1:
            raise NoSuchDesignerError("PPODesigner only supports num_arms=1")

        t0 = time.time()
        policy = data[-1].policy.clone() if data else self._policy.clone()
        self._validate_policy(policy)

        traj = collect_trajectory(self._env_conf, policy)
        if traj.values is None or traj.rewards is None or traj.dones is None or traj.log_probs is None:
            raise NoSuchDesignerError("Trajectory missing required PPO data (values, rewards, dones, log_probs)")

        cfg = self._config
        advantages, returns = compute_gae(traj.rewards, traj.values, traj.dones, cfg.gamma, cfg.gae_lambda)

        device = next(policy.parameters()).device
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        batch = _PPOBatch(
            obs=torch.as_tensor(traj.states.T, dtype=torch.float32, device=device),
            actions=torch.as_tensor(traj.actions.T, dtype=torch.float32, device=device),
            old_log_probs=torch.as_tensor(traj.log_probs, dtype=torch.float32, device=device),
            advantages=(adv_t - adv_t.mean()) / (adv_t.std() + 1e-8),
            returns=torch.as_tensor(returns, dtype=torch.float32, device=device),
        )

        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        for _ in range(cfg.epochs):
            _ppo_update_epoch(policy, optimizer, batch, cfg)

        if telemetry is not None:
            telemetry.set_dt_fit(time.time() - t0)
            telemetry.set_dt_select(0.0)

        return [policy]
