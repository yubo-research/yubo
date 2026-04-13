
## Scope

`mehul/TR` extends the BO stack around `turbo-enn-*` with:

- non-box trust-region geometries
- module-aware perturbation for neural-network policies
- richer geometry updates from the ENN surrogate

The added surface supports more structured local search than plain box TuRBO while keeping the existing BO experiment entrypoint.

## Main New Additions

### 1. Trust-region geometries

The main added public surface is the `geometry` option on `turbo-enn-*` configs.

Supported geometries:

- default omitted geometry: plain box trust region
- `enn_iso`
- `enn_metr`
- `grad_metr`
- `enn_ellip`
- `grad_ellip`

Intended meaning:

- `enn_iso`
  - isotropic control
  - identity-metric baseline in the metric-shaped codepath
- `enn_metr`
  - learned metric-shaped trust region
- `grad_metr`
  - learned metric-shaped trust region using gradient signal when available
- `enn_ellip`
  - true ellipsoidal trust region
- `grad_ellip`
  - true ellipsoidal trust region with gradient signal

Semantic rule:

- omitting `geometry` means box
- non-box geometry is the override case

Relevant files:

- [`optimizer/trust_region_config.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/trust_region_config.py)
- [`optimizer/metric_trust_region.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/metric_trust_region.py)
- [`optimizer/ellipsoidal_trust_region.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/ellipsoidal_trust_region.py)
- [`optimizer/trust_region_utils.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/trust_region_utils.py)

### 2. Accelerated trust-region math

`mehul/TR` adds an accel layer for trust-region math only. The public surface is `use_accel` plus optional `accel` selection. Backends are `mlx`, `triton`, `jax`, with NumPy fallback.

[`optimizer/trust_region_accel.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/trust_region_accel.py) is the backend-neutral entrypoint. It:

- selects the active backend in priority order
- honors `use_accel`, `accel`, and `YUBO_TR_ACCEL`
- exposes a single method surface to the trust-region code
- caches covariance factorizations in `CovCache`
- falls back to NumPy when no accel backend is active

Core invariant:

- if `use_accel = false`, trust-region math stays on NumPy
- non-accel paths must not pre-consume Sobol samples from accel-only fast paths

[`optimizer/trust_region_ops.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/trust_region_ops.py) holds the shared formulas behind a small `BackendOps` protocol. The method surface is grouped into four families:

- quadratic-form methods:
  - `mahalanobis_sq_formula`
  - `mahalanobis_from_cov_formula`
  - `mahalanobis_from_factor_formula`
- low-rank methods:
  - `low_rank_step_formula`
  - `low_rank_step_with_sparse_formula`
  - `low_rank_metric_formula`
- sampling and projection methods:
  - `clip_step_formula`
  - `whitened_sample_formula`
- fused candidate builders:
  - `fused_metric_candidates_formula`
  - `fused_low_rank_candidates_formula`
  - `fused_ellipsoid_generate_formula`
  - `fused_whitened_ellipsoid_candidates_formula`

The trust-region semantics are defined in the shared formulas once. Backend adapters differ only in execution substrate, compilation strategy, and optional kernel specialization.

Backend-specific implementation:

- MLX: [`optimizer/trust_region_accel_mlx.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/trust_region_accel_mlx.py)
  - executes the shared formulas on MLX arrays
  - forces execution with `mx.eval(...)` before converting back to NumPy
  - uses MLX linear algebra for factorization and solves, with some covariance solves routed to `stream=mx.cpu`
- JAX: [`optimizer/trust_region_accel_jax.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/trust_region_accel_jax.py)
  - wraps the shared formulas in `jit`-compiled kernels cached through `lru_cache`
  - passes scalar specializations such as the boundary flag and inverse dimension to reduce recompilation pressure
  - computes `mahalanobis_sq_from_cov(...)` through factorization and then reuses the factor-form kernel
- Triton: [`optimizer/trust_region_accel_triton.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/trust_region_accel_triton.py)
  - uses PyTorch CUDA tensors as the array type
  - runs the shared formulas through PyTorch ops by default
  - replaces selected hot operations with Triton kernels for dense Mahalanobis multiplication, matrix multiply, and low-rank metric evaluation
  - short-circuits the rank-zero low-rank metric case to the isotropic closed form instead of compiling a degenerate Triton kernel

### 3. Module-aware perturbation

`mehul/TR` also adds module-aware candidate perturbation for `turbo-enn-*`.

Public knobs:

- `module_tr`
- `module_tr_block_prob`
- `module_tr_min_num_params`

What it does:

- discovers leaf-module parameter blocks in the policy
- perturbs blocks together instead of treating all scalars independently

This is the submodule perturbator path.

Semantic rule:

- `module_tr` is a `turbo-enn-*` feature
- it is not a generic box-trust-region feature in the public config surface
- the box sampler is one internal reuse point

Relevant files:

- [`optimizer/submodule_perturbator.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/submodule_perturbator.py)
- [`optimizer/box_trust_region.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/box_trust_region.py)
- [`optimizer/turbo_enn_designer_ext.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/turbo_enn_designer_ext.py)
- [`optimizer/trust_region_sampling_utils.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/trust_region_sampling_utils.py)

### 4. ENN-driven local geometry updates

The branch extends the ENN surrogate so the trust region can observe:

- local neighborhood geometry
- gradient-derived local geometry
- incumbent transition behavior for ellipsoidal radius updates
- optional PC-rotation geometry

Relevant files:

- [`optimizer/enn_surrogate_ext.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/enn_surrogate_ext.py)

## Main Execution Path

The BO entrypoint is still:

```bash
micromamba run -n yubo-rl python -m experiments.experiment local <config>
```

For Turbo-ENN BO:

1. config is parsed through [`optimizer/designer_registry.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/designer_registry.py)
2. designer is built via [`optimizer/turbo_enn_designer.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/turbo_enn_designer.py) or [`optimizer/turbo_enn_designer_ext.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/turbo_enn_designer_ext.py)
3. the trust region is created from [`optimizer/trust_region_config.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/trust_region_config.py)
4. the optimizer ask path runs through [`optimizer/enn_turbo_optimizer.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/enn_turbo_optimizer.py)
5. the ENN surrogate updates local geometry via [`optimizer/enn_surrogate_ext.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/enn_surrogate_ext.py)

The high-level per-round sequence is:

1. fit/update ENN surrogate
2. update trust-region geometry near the incumbent
3. generate candidate set inside the current trust region
4. score candidates with Thompson / UCB / Pareto acquisition
5. evaluate selected arms in the environment
6. update trust-region state

## Config Semantics

### Geometry

- omit `geometry` for default box behavior
- set `geometry` only for non-box trust-region overrides

### Covmat

For metric/ellipsoid geometry:

- `covmat = "low_rank"` gives lower proposal-time cost
- `covmat = "dense"` gives denser geometry and higher cost

### Ellipsoidal-only parameter

- `radial_mode` only matters for true ellipsoid geometry
- it is not meaningful for `enn_metr` / `grad_metr`

Current intended values:

- `ball_uniform`
- `boundary`

### Accel

- `use_accel = true` means the trust-region math may use `mlx` / `triton` / `jax`
- `use_accel = false` must force NumPy behavior for the trust-region math

### Module perturbation

- `module_tr = true` enables submodule perturbation
- this is a proposal-distribution change, not a new objective

## Computational Complexity

Use the following notation:

- `d`: policy parameter dimension
- `N_t`: number of BO observations at round `t`
- `M`: `num_candidates`
- `k`: ENN neighborhood size
- `P`: `num_fit_samples`
- `C`: `num_fit_candidates`
- `r`: low-rank trust-region rank
- `A`: `num_arms`
- `m`: number of objectives

### ENN surrogate fit

Current behavior with the default flat ENN index is approximately:

\[
T_{\text{fit}}(t)=O(PN_t d + CPk m)
\]

### Candidate scoring

For `M` candidates, ENN scoring is dominated by neighbor lookup:

\[
T_{\text{score}}(t)=O(MN_t d + Mk m)
\]

### Candidate generation by geometry

- box / module-aware box:
  - `O(Md)`
- `enn_iso`:
  - `O(Md)`
- low-rank metric / low-rank ellipsoid:
  - `O(Mdr)`
- dense metric / dense ellipsoid:
  - `O(Md^2)`

### Geometry update

- low-rank geometry update:
  - roughly `O(N_t d + min(k d^2, d k^2))`
- dense geometry update:
  - roughly `O(N_t d + k d^2 + d^3)`

Implication:

- low-rank methods have lower asymptotic cost and better scaling in `d`
- dense ellipsoid / dense metric methods are the expensive geometric variants



## Invariants

1. `use_accel = false` must keep the trust-region math on NumPy.

2. Non-accel paths must not pre-consume Sobol samples from accel-only fast paths.

3. Omitting `geometry` means box. Explicit box is not the main public surface for `turbo-enn-*`.

4. `module_tr` is a Turbo-ENN feature, not generic box branding.

5. `radial_mode` is ellipsoid-only.

6. Dense and low-rank paths must preserve the same trust-region containment semantics.

7. The trust-region backend wrappers may change constants, but they must not silently change geometry semantics.

## Representative Configs

### Metric trust region

- [`configs/bo/gymnasium/cheetah/exp_cheetah_metric_thompson_1m.toml`](/Users/mehulbafna/Desktop/DS/yubo-tr/configs/bo/gymnasium/cheetah/exp_cheetah_metric_thompson_1m.toml)

Run:

```bash
cd /Users/mehulbafna/Desktop/DS/yubo-tr
micromamba run -n yubo-rl python -m experiments.experiment local \
  configs/bo/gymnasium/cheetah/exp_cheetah_metric_thompson_1m.toml
```

### Ellipsoidal trust region

- [`configs/bo/gymnasium/cheetah/exp_cheetah_ellip_thompson_1m.toml`](/Users/mehulbafna/Desktop/DS/yubo-tr/configs/bo/gymnasium/cheetah/exp_cheetah_ellip_thompson_1m.toml)

Run:

```bash
cd /Users/mehulbafna/Desktop/DS/yubo-tr
micromamba run -n yubo-rl python -m experiments.experiment local \
  configs/bo/gymnasium/cheetah/exp_cheetah_ellip_thompson_1m.toml
```

### Module-aware box-like baseline

- [`configs/bo/gymnasium/cheetah/exp_cheetah_module_ucb_1m.toml`](/Users/mehulbafna/Desktop/DS/yubo-tr/configs/bo/gymnasium/cheetah/exp_cheetah_module_ucb_1m.toml)

### Module-aware metric variant

- [`configs/bo/gymnasium/cheetah/exp_cheetah_metric_module_thompson_1m.toml`](/Users/mehulbafna/Desktop/DS/yubo-tr/configs/bo/gymnasium/cheetah/exp_cheetah_metric_module_thompson_1m.toml)

## Tests That Matter

The trust-region work is primarily guarded by:

- [`tests/test_metric_shaped_trust_region.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/tests/test_metric_shaped_trust_region.py)
- [`tests/test_trust_region_utils.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/tests/test_trust_region_utils.py)
- [`tests/test_trust_region_jax.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/tests/test_trust_region_jax.py)
- [`tests/test_designer_registry.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/tests/test_designer_registry.py)
- [`tests/test_enn_surrogate_ext.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/tests/test_enn_surrogate_ext.py)

Validation slice:

```bash
cd /Users/mehulbafna/Desktop/DS/yubo-tr
micromamba run -n yubo-rl python -m pytest -q \
  tests/test_metric_shaped_trust_region.py \
  tests/test_trust_region_utils.py \
  tests/test_trust_region_jax.py
micromamba run -n yubo-rl kiss check .
micromamba run -n yubo-rl pre-commit run ruff --all-files
micromamba run -n yubo-rl pre-commit run ruff-format --all-files
```

## Other Additions

`mehul/TR` also includes:

- multi-objective Turbo-ENN support:
  - [`optimizer/multi_turbo_enn_allocation.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/multi_turbo_enn_allocation.py)
  - [`optimizer/multi_turbo_enn_designer.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/multi_turbo_enn_designer.py)
  - [`optimizer/multi_turbo_enn_scoring.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/multi_turbo_enn_scoring.py)
  - [`optimizer/multi_turbo_enn_state.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/optimizer/multi_turbo_enn_state.py)
- policy additions:
  - [`policies/moe_policy.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/policies/moe_policy.py)
  - new registry presets in [`policies/registry.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/policies/registry.py)
- video/replay cleanup:
  - [`common/video.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/common/video.py)
  - [`experiments/experiment_sampler.py`](/Users/mehulbafna/Desktop/DS/yubo-tr/experiments/experiment_sampler.py)

These changes are outside the core trust-region geometry work.

## Summary

- the main addition is non-box local geometry with optional acceleration
- `module_tr` is structured candidate perturbation for neural policies
- the expensive variants are the dense metric / dense ellipsoid paths
- the most sensitive invariants are accel gating, Sobol consumption, and trust-region containment
