# Yubo Branch Summary

This doc is a concise snapshot of this branch and may change as the branch evolves.
Scope: `yubo/` only.

## What Changed (high level, subject to change)
- **TOML configs**: Added a TOML loader + runner (`common/config_toml.py`, `experiments/experiment_toml.py`) so BO/RL runs can be config driven. Currently scoped to TuRBO‑ENN experiments.
- **TuRBO‑ENN integration**: New ENN/TuRBO optimizer stack with ellipsoid trust‑region support and multi‑region designer utilities.
- **Checkpointing/resume**: Optimizer‑level checkpointing utilities for long BO runs.
- **RL backbone + PPO wiring**: Backbone abstractions and PPO runner paths for policy training and evaluation.
- **Backbone/hidden size parametrization**: Direction is fewer knobs in config files via better abstractions; expect changes here.
- **Tests**: Expanded unit tests for configs, designers, optimizers, and RL utilities.
- **Example configs**: Added canonical TOML examples for BO (TuRBO‑ENN) and PPO.

## Key New/Updated Modules
- `common/config_toml.py`: load/apply TOML overrides.
- `experiments/experiment_toml.py`: TOML entrypoint, builds `opt_name` from params.
- `optimizer/enn_turbo_optimizer.py`: ENN‑aware TuRBO optimizer wrapper.
- `optimizer/ellipsoid_trust_region.py`: ellipsoidal trust‑region geometry.
- `optimizer/multi_turbo_enn_designer.py`: multi‑region TuRBO‑ENN strategy.
- `optimizer/checkpointing.py`: optimizer state save/load helpers.
- `rl/policy_backbone.py`, `rl/backbone.py`, `rl/actor_critic.py`: backbone/actor‑critic wiring.
- `rl/algos/ppo.py`, `rl/algos/runner.py`: PPO run path.

## How to Author Experiment TOMLs (no new CLI)
Use existing examples as your base (reference; may change):
- BO/TuRBO‑ENN: `experiments/configs/halfcheetah_turbo_enn_fit_ellipsoid.toml`
- PPO/RL: `experiments/configs/pendulum_ppo.toml`

### BO / Example of a TuRBO‑ENN TOML config file
Minimum structure:
```
[experiment]
exp_dir = "_tmp/your_run_dir"
env_tag = "cheetah-mlp"
num_arms = 10
num_rounds = 10000
num_reps = 1
num_denoise = 1
num_denoise_passive = 1
b_trace = true

[optimizer]
name = "turbo-enn-fit"

[optimizer.params]
acq_type = "thompson"           # or "ucb", "pareto"
geometry = "enn_ellipsoid"      # or "box"
sampler = "full"                # "full" or "low_rank"

[optimizer.general]
num_keep = 10
keep_style = "best"
```
Notes (concise):
- `experiment_toml.py` **builds** `opt_name` from `[optimizer.params]` + `[optimizer.general]`.
- Use `[experiment].run_workers` if you want parallel runs (keeps behavior consistent with existing code).

### PPO / RL TOML shape (minimal)
Minimum structure:
```
[rl]
algo = "ppo"

[rl.ppo]
exp_dir = "_tmp/ppo_run"
env_tag = "pend"
seed = 1
total_timesteps = 200000
num_envs = 1
num_steps = 1024
update_epochs = 10
num_minibatches = 32
learning_rate = 3e-4

backbone_name = "mlp"
backbone_hidden_sizes = [64, 64]
backbone_activation = "silu"
backbone_layer_norm = true
share_backbone = true

[rl.run]
seeds = [1,2,3,4,5,6,7,8,9,10]
workers = 10
```
Notes (concise):
- `rl.run` controls multi‑seed execution.
- Backbone settings are centralized here to keep PPO and evaluation consistent.

## Suggested Baseline Examples (reference)
- Ellipsoid TuRBO‑ENN: `experiments/configs/halfcheetah_turbo_enn_fit_ellipsoid.toml`
- Multi‑region stress test: `experiments/configs/sphere_2m_turbo_enn_multi_uniform.toml`
- PPO sanity check: `experiments/configs/pendulum_ppo.toml`

## Running TOML configs (concise)
- If the `yubo` env is already active (repo root): `python -m experiments.experiment_toml --config path/to/config.toml`
- One‑liner (no activation step): `micromamba run -n yubo python -m experiments.experiment_toml --config path/to/config.toml`
- TOML can live anywhere (including `_tmp/`), as long as `--config` points to it.
- If the environment needs updates, follow `README.md` setup and re‑install deps (e.g., `requirements.txt`).
