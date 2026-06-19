# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

The project targets **pixi** (`pixi.toml`, environment `hyperscalees`) for the full stack. On this Mac the active install is a **micromamba** env at `/Users/vinodkumar/micromamba/envs/yubo` which is a partial install.

```bash
# Full setup (Apple Silicon) — installs everything including ennbo, torchrl, jax
pixi install -e hyperscalees
pixi run -e hyperscalees bootstrap-mac   # or: extras-mac && check-mac
```

If pixi is not yet installed, the micromamba env works for BO experiments only (see "What runs where" below). Always run CLIs from the repo root with `PYTHONPATH` set:

```bash
# Without pixi (micromamba env)
export PYTHONPATH=$(pwd)
/Users/vinodkumar/micromamba/envs/yubo/bin/python ops/experiment.py local configs/basic.toml

# With pixi
pixi run -e hyperscalees ./ops/experiment.py local configs/basic.toml
```

All `./ops/....py` scripts are shebang scripts; run them from the repository root.

### What runs where

| Experiment type | micromamba `yubo` | pixi `hyperscalees` |
|---|---|---|
| BO (`ops/experiment.py`) with GPyTorch designers (turbo-1, cma, lhd, ei, ucb, cma, optuna, ax …) | ✓ | ✓ |
| BO with ENN designers (turbo-enn-*, morbo-enn-*) | ✗ needs `ennbo` | ✓ |
| UHD (`ops/exp_uhd.py`) | ✗ needs `ennbo` | ✓ |
| RL SAC/PPO (`ops/rl.py`) | ✗ needs `torchrl`/`tensordict` | ✓ |
| MJX/JAX environments | ✗ needs `jax` | ✓ (CUDA) |
| Modal cloud dispatch | ✓ (`modal` installed) | ✓ |

## Commands

```bash
# One-time setup: install pre-commit hooks
pre-commit install

# Lint, format, and run quality gates (ruff + kiss, config at admin/ruff.toml)
pre-commit run --all-files

# Run all tests (pytest-testmon: only reruns tests affected by changed files)
pytest -sv tests

# Run a single test file
pytest -sv tests/test_<name>.py

# Quality gate used by pre-commit
kiss check

# Show which tests cover a specific file (iterate while fixing)
kiss show-tests path/to/file.py
```

## ops/ CLI reference (run from repo root)

All CLIs are documented in `docs/doc.pdf`. Summary:

### UHD experiments (`ops/exp_uhd.py`)

```bash
# Single run, local
./ops/exp_uhd.py local path/to/config.toml

# With config key overrides
./ops/exp_uhd.py local path/to/config.toml -o num_rounds=50 -o lr=0.01

# Single run on Modal (GPU names: T4, A10, A100, H100)
./ops/exp_uhd.py modal path/to/config.toml [--log-file run.log] [--gpu A100]

# ops/exp_uhd_full.py is a variant with eager imports; same interface
```

### Repeated UHD runs (`ops/uhd_batch.py`)

```bash
# Local: threads + subprocesses, writes traces under results/uhd/
./ops/uhd_batch.py local config.toml --num-reps N [--workers W] [--results-dir DIR]

# Modal: single config, many reps
./ops/uhd_batch.py modal config.toml --num-reps N [--results-dir DIR]

# Modal: many configs from a prep function
./ops/uhd_batch.py batch module.path.prep_fn [--results-dir DIR]

# Result management subcommands
./ops/uhd_batch.py collect [--results-dir DIR]
./ops/uhd_batch.py status [--results-dir DIR]
./ops/uhd_batch.py cleanup [--results-dir DIR]
```

### General BO experiment batches (`ops/batches.py`)

```bash
./ops/batches.py BATCH_TAG [--max-parallel P] [--dry-run] [--results-dir results]
```

`BATCH_TAG` names a function in `experiments/batch_preps.py` (prepends `prep_` if missing). Each `prep_*(results_dir)` returns a list of `ExperimentConfig` objects.

### General BO experiment runner (`ops/experiment.py`)

```bash
./ops/experiment.py local path/to/config.toml
./ops/experiment.py modal path/to/config.toml
```

Forwards all arguments to `experiments.experiment` Click CLI.

### RL runner (`ops/rl.py`)

```bash
./ops/rl.py local configs/rl/gymnasium/cheetah/sac_torchrl_halfcheetah_sota_like_macos.toml
```

Equivalent to passing `--config <path>` to `rl.runner.main`.

### Introspection and data tools

```bash
# List valid names before editing configs
./ops/catalog.py designers       # BO designer base names + options
./ops/catalog.py policies        # policy_tag values
./ops/catalog.py environments    # env_tag values
./ops/catalog.py uhd             # UHD optimizer names (simple, simple_be, mezo, mezo_be, bszo)
./ops/catalog.py jax-envs        # JAX env family tags
./ops/catalog.py llm-envs        # LLM env tags
./ops/catalog.py pretrain-envs   # Pre-training env tags
./ops/catalog.py rl-algos        # RL algorithm names
./ops/catalog.py rl-configs      # Paths of all RL TOML configs

# Inspect UHD batch result trees
./ops/data.py ls   [--results-dir DIR]      # list experiment subdirs
./ops/data.py cat  [--results-dir DIR]      # print config + trace JSONL
./ops/data.py rm   HASH [--results-dir DIR] # delete by hash

# Quick env sanity check (approximate noise, mean return, param count)
python ops/about_env.py ENV_TAG POLICY_TAG

# MNIST training demo
./ops/fit_mnist.py [--epochs N] [--batch-size B] [--lr LR] [--timeout T]
```

### Modal batch wrappers

```bash
# modal_batches: deploy/submit/collect/stop for experiment batches
./ops/modal_batches.py deploy TAG
./ops/modal_batches.py submit TAG BATCH_TAG
./ops/modal_batches.py collect TAG
./ops/modal_batches.py status TAG
./ops/modal_batches.py stop TAG

# Timing sweep
./ops/single_run_time.py deploy
./ops/single_run_time.py submit BATCH_TAG --prep experiments.batch_preps.prep_...
./ops/single_run_time.py progress --prep ...
./ops/single_run_time.py collect

# Synthetic sine surrogate benchmark
./ops/synthetic_sine_benchmark_batches.py [deploy|submit|collect|stop] [--help]
```

## Config schemas

### BO experiment config (`ops/experiment.py`)

Used with `./ops/experiment.py local` and `ops/batches.py`. Config can live at root or under `[experiment]`. An `[optimizer]` table can replace `opt_name`:

```toml
[experiment]
exp_dir     = "runs/my_run"   # required
env_tag     = "cheetah"       # required — see `./ops/catalog.py environments`
policy_tag  = "linear"        # required — see `./ops/catalog.py policies`
num_arms    = 2               # required
num_reps    = 10              # required
num_rounds  = 10000           # required (or total_timesteps)
num_denoise = 1               # optional
num_denoise_passive = 1       # optional
b_trace     = true            # optional
runtime_device = "cpu"        # optional
local_workers = 10            # optional

[optimizer]
name = "turbo-enn-fit-ucb"    # from `./ops/catalog.py designers`
# [optimizer.params]          # optional per-designer params
```

For EggRoll optimizers, use `population` instead of `num_arms` and `num_epochs` instead of `num_rounds`; they are auto-mapped.

### UHD config (`ops/exp_uhd.py`)

Config lives at root or under `[uhd]`. Parsed by `ops/exp_uhd_parse.py`.

**Required:** `env_tag` and one of `num_rounds` / `total_timesteps`.

```toml
[uhd]
env_tag    = "mnist_fulltrain_acc"   # required
policy_tag = "linear"                # optional
num_rounds = 1000                    # required (or total_timesteps for gymnax: envs)

# Core optimizer
optimizer  = "mezo"        # simple | simple_be | mezo | mezo_be | bszo | bszo_be (default: mezo)
lr         = 0.001         # default 0.001
sigma      = 0.001         # perturbation scale (default 0.001)
perturb    = "dim:0.5"     # dense | dim:<n> | mod:<n> | eggroll (default: dim:0.5)
batch_size = 4096          # default 4096
log_interval      = 1
accuracy_interval = 1000
target_accuracy   = 0.9    # stop early when reached

# Replication (when using uhd_batch.py)
num_reps    = 5
seed_offset = 0

# BehavioralEmbedder (optimizer = simple_be / mezo_be)
be_num_probes     = 10
be_num_candidates = 10
be_warmup         = 20
be_fit_interval   = 10
be_enn_k          = 25
be_acquisition    = "ucb"   # ucb | ts | ...

# ENN surrogate imputation (enn_minus_impute = true)
enn_minus_impute    = false
enn_d               = 100    # JL embedding dim
enn_s               = 4      # JL sparsity
enn_jl_seed         = 123
enn_k               = 25
enn_fit_interval    = 50
enn_warmup_real_obs = 200
enn_se_threshold    = 0.25

# Early-reject
er_tau      = 0.1
er_mode     = "ema"   # ema | quantile | ...
er_ema_beta = 0.9

# BSZO (optimizer = bszo / bszo_be)
bszo_k           = 2
bszo_epsilon     = 1e-4
bszo_sigma_p_sq  = 1.0
bszo_sigma_e_sq  = 1.0
bszo_alpha       = 0.1
```

### RL config (`ops/rl.py`)

Config must contain `[rl]` with `algo = "ppo" | "sac" | "mjx_ppo" | "mjx_sac"`. Algorithm-specific settings go under `[rl.<algo>]`.

```toml
[rl]
algo = "sac"

[rl.run]
num_reps = 10
workers  = 1

[rl.run.artifacts.video]
enable = true

[rl.sac]
exp_dir    = "runs/rl/my_run"
env_tag    = "cheetah"
policy_tag = "mlp-32-16"
device     = "cpu"           # cpu on macOS (MPS not supported for RL sim)
log_interval_steps = 1000

[rl.sac.collector]
total_frames       = 1000000
num_envs           = 1
frames_per_batch   = 32
init_random_frames = 10000
backend            = "single"

[rl.sac.replay_buffer]
size       = 1000000
batch_size = 256

[rl.sac.optim]
actor_lr   = 3e-4
qvalue_lr  = 3e-4
alpha_lr   = 3e-4

[rl.sac.loss]
gamma          = 0.99
alpha_init     = 0.2
target_entropy = -6.0

[rl.sac.target_net_updater]
tau = 0.005

[rl.sac.eval]
interval_steps      = 5000
num_denoise         = 1
num_denoise_passive = 1
noise_mode          = "frozen"

[rl.sac.checkpoint]
interval_steps = 100000
```

For MJX (JAX-based) PPO on CUDA:
```toml
[rl]
algo = "mjx_ppo"

[rl.mjx_ppo]
exp_dir     = "runs/rl/mjx/run"
env_tag     = "gymnasium_fast:HalfCheetah-v5"
seed        = 0
hidden_size = 256

[rl.mjx_ppo.collector]
total_frames = 5000000
num_envs     = 2048
num_steps    = 32
```

## Platform constraints

- **macOS (MPS):** CPU-only for RL simulation and GPU-sim environments (IsaacLab, MJX with CUDA, `gymnasium_fast:*` envs require CUDA). Use `device = "cpu"` in RL configs on macOS.
- **Modal:** Use for CUDA workloads (`--gpu A100` etc.). GPU names follow Modal device strings: T4, A10, A100, H100.
- **ennbo:** PyPI name for the ENN library. Installed via `admin/install.sh` or `admin/install-macos.sh`. The Modal image for UHD runs also builds the sibling `../enn` checkout with maturin.

## Architecture

### Top-level modules

| Module | Purpose |
|--------|---------|
| `acq/` | GP fitting, acquisition functions (Thompson sampling, Turbo, DPP, etc.) |
| `optimizer/` | BO designer implementations and registries |
| `problems/` | Objective functions and environments (DM Control, MuJoCo/MJX, Atari, IsaacLab, MNIST, LLM fine-tuning) |
| `experiments/` | Experiment orchestration: local runs, Modal cloud dispatch, batch jobs |
| `ops/` | CLI entry points that forward to `experiments/` |
| `sampling/` | Sampling utilities: MCMC, Sobol, sparse JL projections, covariance tools |
| `rl/` | RL algorithms: PPO and SAC, MJX (JAX) and TorchRL backends |
| `llm/` | LLM training/eval: vLLM actors, EggRoll RL policy, LoRA, verifiers |
| `model/` | Neural network definitions (MLP, CNN, shared Gaussian actor) |
| `embedding/` | Behavioral embedding utilities |
| `common/` | Config (TOML), console/logging, telemetry, seed management |
| `analysis/` | Result loading, data sets, plotting |

### Core abstractions

**Designer (BO strategy):** Implements `optimizer/designer_protocol.py`. Each designer has `ask()` / `tell()` methods. Registered in `optimizer/designer_registry.py` by string name; the registry defers object construction until a name is selected. Use `./ops/catalog.py designers` to list valid names.

**Problem (objective):** Wraps an environment or benchmark function. Lives in `problems/`. `env_conf.py` parses TOML config into environment + policy specs.

**UHD (Ultra-High-Dimensional) optimization:** `optimizer/uhd_*.py`. Perturbation-based BO for very high-dimensional parameter spaces. UHD perturbators **must not allocate O(D) memory** — apply perturbations in-place or with sparse-index updates. See `grounding.md`.

**UHDMeZO algorithm:** Antithetic zeroth-order gradient — evaluates at `x+ = x + σε` and `x- = x - σε`, computes projected gradient `g_proj = (μ+ - μ-) / 2σ`, applies Adam-style scalar normalization. No D-sized noise vector is stored; perturbators regenerate noise from seed.

**ENN (Ensemble Neural Network):** Surrogate model providing posterior mean `μ(x)` and calibrated epistemic uncertainty `se(x)`. Used via `enn_fit()` + `posterior()`. PyPI package name: `ennbo`. Integration in `optimizer/uhd_enn.py` and `analysis/fitting_time/`.

**SparseJL / DeltaSparseJL:** `sampling/sparse_jl_t.py` and `sampling/delta_sparse_jl_t.py`. Linear embedding `T: R^D → R^d` with O(1) memory (chunked, no D-sized allocations). Key APIs:
- `block_sparse_jl_transform_t(x, d, s=4, seed=42)` — tensor input
- `block_sparse_jl_transform_module(module, d, s=4, seed=42)` — iterate module params without flattening
- `DeltaSparseJL_T` — incremental embedding updates for sparse deltas

**EggRoll:** RL-trained BO policy. Uses `population` / `num_epochs` in TOML instead of `num_arms` / `num_rounds`; auto-mapped in `experiments/experiment.py`.

**Modal:** Cloud compute backend. Image definitions live in `experiments/modal_*.py`. `MODAL_TAG` env var isolates concurrent deployments.

### Noise conventions (`grounding.md`)

- **Frozen noise:** For denoised evaluation with `num_denoise = k`, always reuse the same k seeds (`noise_seed_0 + offset + 0..k-1`) each time the objective is evaluated.
- **Natural noise:** Optimization-feedback evaluations use fresh seeds; held-out evaluative metrics use a fixed `num_denoise_passive` seed set.

### Test layout

All tests live flat in `tests/`. Two categories:
- `test_*.py` — actual pytest test files
- `kiss_*.py`, `*_helpers.py`, `*_stubs.py`, `*_support.py` — fixtures and stubs imported by test files

`tests/conftest.py` appends (not prepends) `tests/` to `sys.path`; repo root must win for `ops.*` imports.

`sitecustomize.py` at repo root patches NumPy legacy aliases (`np.NaN`, `np.Inf`) and is imported automatically during test runs.

## Quality gates (kiss)

`kiss check` runs as a pre-commit hook. Key thresholds from `.kissconfig`:

- `lines_per_file`: 500 — split files before crossing this
- `statements_per_file`: 300 — statements inside function bodies
- `statements_per_function`: 50
- `branches_per_function`: 9 — use `_HANDLER` dispatch tables when nearing this limit
- `test_coverage_threshold`: 90% — every new module needs kiss coverage tests
- `duplication_enabled`: true — fix with shared helpers, not copy-paste

When `calls_per_function` or `local_variables` limits are hit, extract helpers into the same package or use `SimpleNamespace` grouping. Use `kiss show-tests <file>` to iterate on a failing file without running the full suite.

## Linting

Ruff config at `admin/ruff.toml`: line-length 160, rules E/F/W + I001 (import sort), `E501` ignored. Pre-commit auto-fixes import order.

Forbidden in committed code: `# TEST`, `# HACK` (blocked by pre-commit hook).
