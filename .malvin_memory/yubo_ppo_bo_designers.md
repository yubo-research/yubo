# Yubo: BO PPO designers (`ppo-ac` / `ppo-pg`)

TRIGGER: ppo-ac ppo-pg registry
ADVICE: Optimizer registry keys `"ppo-ac"`→`PPOACDesigner` in `optimizer/ppo_designer.py` (needs `last_values` / actor-critic policy) and `"ppo-pg"`→`PPOPGDesigner` in `optimizer/ppo_pg_designer.py` with `ActorMLPPolicy` (`policies/actor_mlp_policy.py`). Shared merge/GAE/advantages in `optimizer/ppo_common.py`. No `opt_name = "ppo"`.
CONFIDENCE: 3

TRIGGER: merge_trajectories num_arms
ADVICE: When merging multiple rollouts (`num_arms > 1`), `merge_trajectories` forces `done=True` on each arm's last timestep so episode-return advantages and GAE do not span arms. A single trajectory is returned unchanged (no forced done).
CONFIDENCE: 3

TRIGGER: normalize_advantages episode return
ADVICE: `compute_episode_return_advantages` broadcasts one scalar return per episode segment, so the advantage vector is often constant. In `normalize_advantages`: use `std(unbiased=False)`; if `numel == 1` return zeros; if `numel > 1` and `std < 1e-8`, return unnormalized (do not zero out PG signal when `ent_coef=0`).
CONFIDENCE: 3

TRIGGER: collect_trajectory actor-only
ADVICE: `optimizer/trajectories.collect_trajectory` sets `values=None` when the policy has no callable `last_values`; `ppo-pg` allows that, `ppo-ac` rejects it. Value-less multi-arm merge must not call `np.concatenate` on `None` (handled in `ppo_common.merge_trajectories`).
CONFIDENCE: 3

TRIGGER: ppo datum policy test
ADVICE: Designers clone `data[-1].policy` for rollout. To test parent-policy use, capture `pol.get_params()` inside a mocked `collect_trajectory` side_effect before the update mutates the clone; do not compare params after `designer()` returns.
CONFIDENCE: 3

TRIGGER: clear_policy_ppo_cache clone
ADVICE: `PPOACDesigner.__call__` clones policy before rollout/update; `clear_policy_ppo_cache` clears the returned clone. Assert `_last_log_prob`/`_last_value` on `designer(...)[0]`, not the pre-clone reference passed to the constructor.
CONFIDENCE: 3

TRIGGER: kiss ppo designer split
ADVICE: Kiss `concrete_types_per_file` allows at most two classes per file; keep `PPOPGDesigner`/`PPOPGConfig` in `ppo_pg_designer.py`, `PPOACDesigner`/`PPOConfig` in `ppo_designer.py`, shared batch/update helpers in `ppo_common.py`. Dedup tanh-Gaussian log-prob via `rl.math_utils.tanh_gaussian_action_log_prob_entropy`.
CONFIDENCE: 3

TRIGGER: malvin plan review LGTM
ADVICE: Read-only KPop plan review: falsify by reading code only; write exactly `LGTM` to `_malvin/<run_id>/review.md` if acceptable, else brief blocking notes. Log hypotheses to `_malvin/<run_id>/_kpop/exp_log_<run_id>.md`.
CONFIDENCE: 3
