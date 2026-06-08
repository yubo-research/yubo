import time
from dataclasses import dataclass

import torch

from optimizer.designer_errors import NoSuchDesignerError
from optimizer.designer_protocol import Designer
from optimizer.ppo_common import (
    _ppo_ac_update_epoch,
    _PPOACBatch,
    apply_ppo_telemetry,
    clear_policy_ppo_cache,
    compute_episode_return_advantages,
    compute_gae,
    merge_trajectories,
    normalize_advantages,
    resolve_designer_config,
    trajectory_tensors,
)
from optimizer.ppo_pg_designer import PPOPGConfig, PPOPGDesigner
from optimizer.trajectories import collect_trajectory

__all__ = [
    "PPOACDesigner",
    "PPOConfig",
    "PPODesigner",
    "PPOPGDesigner",
    "PPOPGConfig",
    "compute_episode_return_advantages",
    "compute_gae",
    "merge_trajectories",
]


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


class PPOACDesigner(Designer):
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
        self._config = resolve_designer_config(config, PPOConfig, overrides)

    def _validate_policy(self, policy) -> None:
        required = ["get_action_and_value", "last_log_probs", "last_values"]
        missing = [m for m in required if not (hasattr(policy, m) and callable(getattr(policy, m)))]
        if missing:
            raise NoSuchDesignerError(
                f"ppo-ac requires a policy with {', '.join(required)} methods "
                f"(missing: {', '.join(missing)}). Use ActorCriticMLPPolicy (e.g., env_tag='lunar-ac')."
            )

    def __call__(self, data, num_arms, *, telemetry=None):
        if num_arms < 1:
            raise NoSuchDesignerError("PPOACDesigner requires num_arms >= 1")

        policy = data[-1].policy.clone() if data else self._policy.clone()
        self._validate_policy(policy)

        t_rollout_start = time.time()
        trajectories = []
        for _ in range(num_arms):
            traj = collect_trajectory(self._env_conf, policy)
            if traj.values is None or traj.rewards is None or traj.dones is None or traj.log_probs is None:
                raise NoSuchDesignerError("Trajectory missing required PPO data (values, rewards, dones, log_probs)")
            trajectories.append(traj)
        dt_rollout = time.time() - t_rollout_start

        cfg = self._config
        merged_traj = merge_trajectories(trajectories)
        advantages, returns = compute_gae(
            merged_traj.rewards,
            merged_traj.values,
            merged_traj.dones,
            cfg.gamma,
            cfg.gae_lambda,
        )

        device = next(policy.parameters()).device
        obs, actions, old_log_probs = trajectory_tensors(merged_traj, device)
        batch = _PPOACBatch(
            obs=obs,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=normalize_advantages(advantages, device),
            returns=torch.as_tensor(returns, dtype=torch.float32, device=device),
        )

        t_update0 = time.time()
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        for _ in range(cfg.epochs):
            _ppo_ac_update_epoch(policy, optimizer, batch, cfg)
        dt_update = time.time() - t_update0

        if telemetry is not None:
            apply_ppo_telemetry(telemetry, dt_rollout, dt_update, num_arms)

        clear_policy_ppo_cache(policy)
        return [policy]


PPODesigner = PPOACDesigner
