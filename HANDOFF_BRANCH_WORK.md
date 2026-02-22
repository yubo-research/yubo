# Branch Handoff: `mehul/rl-etc`

## Scope
Delta against `dsweet/sparse` with two main tracks:
1. BO trust-region upgrades (multi-region + ellipsoidal/metric shaping).
2. RL backend refactor (TorchRL + PufferLib) with reproducible setup/configs.

## BO: Trust-Region Work
Added a full trust-region stack for ENN/TuRBO with explicit geometry + multi-region allocation.

Key modules:
- `optimizer/ellipsoidal_trust_region.py`
- `optimizer/metric_trust_region.py`
- `optimizer/trust_region_config.py`
- `optimizer/trust_region_math.py`
- `optimizer/trust_region_utils.py`
- `optimizer/multi_turbo_enn_allocation.py`
- `optimizer/multi_turbo_enn_scoring.py`
- `optimizer/multi_turbo_enn_state.py`
- `optimizer/multi_turbo_enn_designer.py`
- `optimizer/multi_turbo_enn_utils.py`
- `optimizer/turbo_enn_designer_ext.py`

Intent:
- Support multiple concurrent TRs.
- Support non-box TR geometries (ellipsoidal + metric-shaped).
- Keep TR policy/config explicit via `trust_region_config`.

## RL: Backend/Infra Work
RL code moved to backend-first layout and shared infra extracted.

Key structure:
- `rl/algos/backends/torchrl/...`
- `rl/algos/backends/pufferlib/...`
- `rl/algos/registry.py`, `rl/algos/runner.py`, `rl/algos/runner_helpers.py`
- `common/console.py`, `rl/algos/logger.py`, `rl/algos/checkpointing.py`, `common/video.py`

Notable decisions:
- Runner supports `local --config ...` path.
- PufferLib compatibility uses repo-local `gym` shim (`gym/__init__.py`) + Gymnasium fallback in `rl/algos/pufferlib_compat.py`.

## RL Configs (Intentionally Scoped)
Only these new RL TOMLs are staged:
- `configs/rl/atari/ppo_pong_puffer_cuda.toml`
- `configs/rl/atari/ppo_pong_puffer_macos.toml`
- `configs/rl/gymnasium/bw/ppo_puffer_cuda.toml`
- `configs/rl/gymnasium/bw/ppo_puffer_macos.toml`

Each has checkpoint + video enabled.

## Setup Assets
- `admin/setup-rl.sh`
- `admin/setup-rl-macos.sh`
- `admin/setup-rl-linux.sh`
- `admin/conda-rl.yml`
- `admin/README-rl.md`

Environment name: `yubo-rl`.

## Validation Snapshot
- `micromamba run -n yubo-rl pre-commit run` passes.
- Targeted checks used during fixes:
  - `pytest -q tests/test_rl_runner.py tests/test_pufferlib_compat.py`

## Practical Notes
- Keep TOML scope tight (4 RL configs above).
- Do not weaken lint/KISS checks; fix via structure.
- If running outside repo root, verify `gym` resolution before PufferLib runs.
