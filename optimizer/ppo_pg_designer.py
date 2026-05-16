import time
from dataclasses import dataclass

import torch

from optimizer.designer_errors import NoSuchDesignerError
from optimizer.designer_protocol import Designer
from optimizer.ppo_common import (
    _ppo_pg_update_epoch,
    _PPOPGBatch,
    apply_ppo_telemetry,
    compute_episode_return_advantages,
    merge_trajectories,
    normalize_advantages,
    resolve_designer_config,
    trajectory_tensors,
)
from optimizer.trajectories import collect_trajectory


@dataclass
class PPOPGConfig:
    lr: float = 3e-4
    clip_coef: float = 0.2
    epochs: int = 4
    gamma: float = 1.0
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5


class PPOPGDesigner(Designer):
    def __init__(
        self,
        policy,
        env_conf,
        *,
        config: PPOPGConfig | None = None,
        **overrides,
    ):
        self._policy = policy
        self._env_conf = env_conf
        self._config = resolve_designer_config(config, PPOPGConfig, overrides)

    def _validate_policy(self, policy) -> None:
        required = ["get_action_and_value", "last_log_probs"]
        missing = [m for m in required if not (hasattr(policy, m) and callable(getattr(policy, m)))]
        if missing:
            raise NoSuchDesignerError(
                f"ppo-pg requires a policy with {', '.join(required)} methods "
                f"(missing: {', '.join(missing)}). Use ActorMLPPolicy (e.g., policy_tag='actor-mlp-32-32')."
            )
        if hasattr(policy, "last_values") and callable(getattr(policy, "last_values")):
            raise NoSuchDesignerError("ppo-pg requires an actor-only policy without last_values (use ActorMLPPolicy).")

    def __call__(self, data, num_arms, *, telemetry=None):
        if num_arms < 1:
            raise NoSuchDesignerError("PPOPGDesigner requires num_arms >= 1")

        policy = data[-1].policy.clone() if data else self._policy.clone()
        self._validate_policy(policy)

        t_rollout_start = time.time()
        trajectories = []
        for _ in range(num_arms):
            traj = collect_trajectory(self._env_conf, policy)
            if traj.rewards is None or traj.dones is None or traj.log_probs is None:
                raise NoSuchDesignerError("Trajectory missing required ppo-pg data (rewards, dones, log_probs)")
            trajectories.append(traj)
        dt_rollout = time.time() - t_rollout_start

        cfg = self._config
        merged_traj = merge_trajectories(trajectories)
        advantages = compute_episode_return_advantages(merged_traj.rewards, merged_traj.dones, gamma=cfg.gamma)

        device = next(policy.parameters()).device
        obs, actions, old_log_probs = trajectory_tensors(merged_traj, device)
        batch = _PPOPGBatch(
            obs=obs,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=normalize_advantages(advantages, device),
        )

        t_update0 = time.time()
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        for _ in range(cfg.epochs):
            _ppo_pg_update_epoch(policy, optimizer, batch, cfg)
        dt_update = time.time() - t_update0

        if telemetry is not None:
            apply_ppo_telemetry(telemetry, dt_rollout, dt_update, num_arms)

        return [policy]
