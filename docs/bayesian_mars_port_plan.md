# Bayesian MARS Port Plan

## Restated Problem

Port the Bayesian MARS designer work from the `mehul/TR` line into the current `mehul/tidy` branch, but first review the implementation thoroughly, identify any paper/reference trail behind it, and write the plan that should drive the code changes.

## Source Audit

Claim: the committed `mehul/TR` ref does not contain the MARS implementation. Evidence: `git grep -i mars mehul/TR -- ':!*.ipynb'` returned no matches, while `git rev-parse --verify mehul/TR` resolved to `a62bb655d00e38be6ccaa47d3c6f5d99c5c95c64`.

Claim: the MARS implementation exists as untracked worktree state in `/Users/mehulbafna/Desktop/yubo/yubo-tr`, layered on top of `mehul/TR`. Evidence: `git --git-dir=/Users/mehulbafna/Desktop/yubo/yubo/.git/worktrees/yubo-tr --work-tree=/Users/mehulbafna/Desktop/yubo/yubo-tr ls-files --others --exclude-standard` lists:

- `/Users/mehulbafna/Desktop/yubo/yubo-tr/optimizer/mars_surrogate.py`
- `/Users/mehulbafna/Desktop/yubo/yubo-tr/optimizer/turbo_mars_designer.py`
- `/Users/mehulbafna/Desktop/yubo/yubo-tr/tests/test_mars_surrogate.py`
- `/Users/mehulbafna/Desktop/yubo/yubo-tr/docs/mars_designer_configs.md`

Claim: the untracked implementation is not standalone relative to `mehul/tidy`. Evidence: `optimizer/turbo_mars_designer.py` imports `optimizer.turbo_enn_designer_ext`, `optimizer.mars_surrogate` imports `optimizer.trust_region_math`, and `optimizer/enn_turbo_optimizer.py` imports `optimizer.enn_surrogate_ext`; those TR-extension modules are absent on `mehul/tidy`.

## Reviewed Implementation

Source file: `/Users/mehulbafna/Desktop/yubo/yubo-tr/optimizer/mars_surrogate.py`

- `MarsSurrogateConfig` defines deterministic MARS defaults: 64 max terms, second-order interactions, 8 bootstrap models, active rank 8, trailing observation window 256, feature screen 512, 3 quantile knots per feature, ridge `1e-6`, active samples 256, and spectral clamps for active-subspace geometry.
- `BayesianMarsSurrogateConfig` wraps a MARS basis with Bayesian linear-regression parameters: coefficient priors, optional fixed observation noise, minimum noise, optional inclusion of noise in predictive sigma, basis refresh interval, and optional model-space MCMC settings.
- `_HingeFactor` and `_MarsTerm` encode MARS hinge basis terms and term gradients on active features.
- `_fit_single_mars` builds a screened, quantile-knot MARS basis, selects main effects by correlation with standardized y, optionally adds pairwise interaction terms using residual correlation, then fits ridge coefficients.
- `MarsSurrogate` exposes the ENN surrogate protocol: `fit`, `predict`, `sample`, `get_incumbent_candidate_indices`, `lengthscales`, and `update_trust_region`. Its uncertainty is a bootstrap ensemble standard deviation.
- `_fit_bayesian_mars` fits Bayesian linear coefficients on a fixed MARS basis. It standardizes y, estimates or consumes observation variance, builds posterior precision, solves for posterior mean/covariance, and returns predictive mean/sigma.
- `_fit_bayesian_mars_mcmc` builds a candidate basis pool, runs add/drop model-space MCMC, weights sampled basis structures by frequency, and refits Bayesian linear posteriors for retained models.
- `BayesianMarsSurrogate` model-averages posterior means and variances across sampled basis structures, samples by drawing a retained model and then drawing coefficients from that model posterior, and can update low-rank trust-region geometry from the posterior mean model.
- `ENNMarsGeometrySurrogate` delegates acquisition predictions/samples to ENN while fitting a MARS surrogate only to produce active-subspace trust-region geometry.

Source file: `/Users/mehulbafna/Desktop/yubo/yubo-tr/optimizer/turbo_mars_designer.py`

- `TurboMARSDesigner` subclasses the TR-extension `TurboENNDesigner`, forces Python backend, replaces the surrogate with `MarsSurrogateConfig`, and supports `pareto`, `ucb`, and `thompson`.
- `TurboBayesianMARSDesigner` is analogous but uses `BayesianMarsSurrogateConfig`, sets `use_y_var=True`, and also supports `pareto`, `ucb`, and `thompson`.
- `TurboENNMARSGeometryDesigner` uses a normal ENN acquisition config and replaces only the surrogate with `ENNMarsGeometrySurrogateConfig`.

Source file: `/Users/mehulbafna/Desktop/yubo/yubo-tr/docs/mars_designer_configs.md`

- Exposes `turbo-mars-ucb`, `turbo-mars-pareto`, `turbo-mars-thompson`, `turbo-mars-as-ucb`, `turbo-bmars-ucb`, `turbo-bmars-pareto`, `turbo-bmars-thompson`, `turbo-bmars-as-ucb`, and `turbo-enn-mars-geometry`.
- Documents stable BMARS defaults: Sobol candidates, 8 candidates, first-order compact basis, trailing obs 32, feature screen 32, active samples 32, predictive sigma with observation noise, and MCMC basis sampling with 32 burn-in, 32 post-burn-in steps, thin 4, 16 retained models, pool size 32, term prior 0.125.

Source file: `/Users/mehulbafna/Desktop/yubo/yubo-tr/tests/test_mars_surrogate.py`

- Tests vectorized hinge-basis generation against a reference loop.
- Tests deterministic MARS fit/predict/sample shape and finite bootstrap sigma.
- Tests active low-rank factor construction.
- Tests BMARS posterior prediction, coefficient sampling, y-var use, MCMC model averaging, post-burn-in behavior, and active low-rank geometry.
- Tests ENN acquisition plus MARS geometry via a dummy ENN surrogate.

## Reference Trail

Claim: no MARS source file or MARS doc in the TR worktree gives an explicit DOI, arXiv, or paper citation. Evidence: `rg -i 'paper|doi|arxiv|friedman|denison|mallick|smith|reference|citation'` over the MARS files only found config docs and code identifiers, not bibliographic references.

Hypothesis: the deterministic MARS basis code is inspired by Friedman’s original MARS model, but simplified for BO. Predictions: source uses hinge functions, feature screening, quantile knots, interactions, and linear regression coefficients, but does not implement Friedman’s full forward/backward GCV pruning. Test result: source has `_HingeFactor`, `_MarsTerm`, `_build_main_basis`, `_fit_single_mars`, feature screening, residual interaction scoring, and ridge solve; no GCV pruning code appears.

Hypothesis: the Bayesian variant is closest to Bayesian MARS / Bayesian model averaging over MARS bases, but is a pragmatic local implementation. Predictions: it should use a MARS basis, Gaussian coefficient priors, noise variance, posterior coefficient covariance, model-space MCMC, and model averaging. Test result: `BayesianMarsSurrogateConfig`, `_bayesian_linear_log_marginal`, `_fit_bayesian_mars_mcmc`, and weighted multi-model `predict` match that structure.

Do not include paper citations in code-facing docs until the implementation owner confirms the intended reference. The branch source and docs do not cite a paper, and inferred literature matches are not enough evidence for a source claim.

## Porting Decision

Do not copy the two untracked MARS files directly into `mehul/tidy`.

Reasons:

- `mars_surrogate.py` is 1061 lines, exceeding the current `kiss` line-per-file limit of 500.
- The file contains more than one responsibility: configs, basis terms, deterministic fitting, Bayesian fitting, MCMC, surrogate protocol wrappers, and ENN geometry composition.
- `turbo_mars_designer.py` depends on the TR-extension designer and custom local Turbo optimizer that are absent in `mehul/tidy`.
- The current `mehul/tidy` registry is split across `designer_registry_builders.py`, `designer_registry_option_handlers.py`, `designer_registry_defs.py`, and `designer_registry_simple_table.py`, so the monolithic TR registry snippets must be adapted to the tidy split layout.

Preferred port strategy:

1. Port the minimum TR-extension substrate needed by MARS/BMARS, then port MARS on top of it.
2. Keep the deterministic MARS and BMARS math split into small modules to satisfy `kiss`.
3. Preserve the public designer strings and defaults from `yubo-tr`.
4. Add tests equivalent to `tests/test_mars_surrogate.py`, plus registry and Ackley smoke coverage adapted to current tidy tests.

## Proposed File Layout

Add these runtime files:

- `optimizer/mars_config.py`
  - `MarsSurrogateConfig`
  - `BayesianMarsSurrogateConfig`
  - `ENNMarsGeometrySurrogateConfig`

- `optimizer/mars_basis.py`
  - `_HingeFactor`
  - `_MarsTerm`
  - `_build_main_basis`
  - scalar-y coercion and basis design helpers

- `optimizer/mars_fit.py`
  - `_fit_single_mars`
  - `_FittedMarsModel`
  - feature screening, top-column scoring, ridge solve

- `optimizer/bayesian_mars_fit.py`
  - `_FittedBayesianMarsModel`
  - `_fit_bayesian_mars`
  - posterior prediction/sample helpers

- `optimizer/bayesian_mars_mcmc.py`
  - MCMC basis pool construction
  - add/drop proposal logic
  - log marginal likelihood
  - `_fit_bayesian_mars_mcmc`

- `optimizer/mars_surrogate.py`
  - public protocol classes: `MarsSurrogate`, `BayesianMarsSurrogate`, `ENNMarsGeometrySurrogate`
  - re-export config and basis names needed by tests

- `optimizer/turbo_mars_designer.py`
  - `TurboMARSDesignerConfig`
  - `TurboMARSDesigner`
  - `TurboBayesianMARSDesignerConfig`
  - `TurboBayesianMARSDesigner`
  - `TurboENNMARSGeometryConfig`
  - `TurboENNMARSGeometryDesigner`

Port or adapt these support files from `mehul/TR` before enabling active-subspace MARS variants:

- `optimizer/trust_region_math.py`
- `optimizer/trust_region_sampling_utils.py`
- `optimizer/trust_region_geometry.py`
- `optimizer/trust_region_config.py`
- `optimizer/box_trust_region.py`
- `optimizer/metric_trust_region.py`
- `optimizer/ellipsoidal_trust_region.py` only if preserving ENN ellipsoid support in the same pass
- `optimizer/trust_region_accel.py` plus backend files only if keeping `use_accel=true` behavior
- `optimizer/enn_surrogate_ext.py`
- `optimizer/enn_turbo_optimizer.py`
- `optimizer/turbo_enn_designer_ext.py`

If the port scope must be smaller, the first shippable subset can be box-only `turbo-mars-*` and `turbo-bmars-*`, with `turbo-mars-as-ucb`, `turbo-bmars-as-ucb`, and `turbo-enn-mars-geometry` deferred until the shaped trust-region substrate is present.

## Registry Plan

Adapt the monolithic TR registry snippets into the current split registry:

- `optimizer/designer_registry_option_handlers.py`
  - Add MARS/BMARS option key sets.
  - Add parse helpers for optional numbers, booleans, strings-in-set, candidate RV, fixed length, MARS config, BMARS config, and module-TR config if the TR substrate is ported.
  - Add builders `_d_turbo_mars_ucb`, `_d_turbo_mars_pareto`, `_d_turbo_mars_thompson`, `_d_turbo_mars_as_ucb`, `_d_turbo_bmars_ucb`, `_d_turbo_bmars_pareto`, `_d_turbo_bmars_thompson`, `_d_turbo_bmars_as_ucb`, and `_d_turbo_enn_mars_geometry`.

- `optimizer/designer_registry_defs.py`
  - Add `DesignerDef` entries for the MARS/BMARS designers.
  - Add option specs equivalent to `_mars_option_specs`, `_bmars_option_specs`, and the active-subspace geometry specs.

- `optimizer/designer_registry.py`
  - Import the MARS handler functions and add them to `_TURBO_OPTION_DISPATCH`, or let `DesignerDef` entries provide them if all are represented there.

- `optimizer/designer_registry_simple_table.py`
  - No change unless a no-option alias is intentionally added. Prefer explicit `DesignerDef` entries to keep catalog metadata.

## Testing Plan

Add or adapt:

- `tests/test_mars_surrogate.py`
  - Port the source tests, adjusted for split modules and public re-exports.

- `tests/test_designer_registry.py`
  - Add registry construction tests for all public strings.
  - Verify BMARS stable defaults.
  - Verify active-subspace variants set `geometry == "mars_metr"` and low-rank covmat when the TR substrate is present.

- `tests/test_ackley_optimizer_variants.py`
  - Add a short Ackley smoke for:
    - `turbo-mars-ucb/...small settings...`
    - `turbo-mars-pareto/...small settings...`
    - `turbo-bmars-ucb`
    - active-subspace variants only after shaped TR support is in place.

Fast validation slices:

```bash
kiss test optimizer/mars_surrogate.py
pytest -q tests/test_mars_surrogate.py
pytest -q tests/test_designer_registry.py -k "mars or bmars"
pytest -q tests/test_ackley_optimizer_variants.py -k "mars or bmars"
```

Final gates:

```bash
kiss check
ruff check .
pytest -sv tests
```

## Implementation Order

1. Add split MARS config, basis, fit, BMARS posterior, BMARS MCMC, and public surrogate wrapper modules.
2. Add the local optimizer hook needed to instantiate custom surrogate config classes. If the full TR substrate is not ready, keep this box-only and force Python backend for MARS/BMARS.
3. Add `turbo_mars_designer.py` with box MARS/BMARS first.
4. Wire registry entries and catalog metadata for box MARS/BMARS.
5. Port tests for MARS/BMARS box behavior and run focused tests.
6. Port shaped trust-region substrate needed for `mars_metr` and `ENNMarsGeometrySurrogate`.
7. Enable active-subspace and ENN-MARS-geometry registry entries.
8. Add active-subspace registry tests and Ackley smoke.
9. Run quality gates and fix all failures.

## Risks

- The source implementation is untracked in `yubo-tr`, so it has not been stabilized through branch history.
- BMARS defaults are intentionally tiny for BO speed; changing them could make smoke tests slow or acquisition noisy.
- Active-subspace variants require the custom local Python optimizer and shaped trust-region hooks. Registering them before those hooks exist will produce runtime failures.
- `kiss` will reject direct large-file copying; the source must be split.
- The current implementation does not implement full Friedman forward/backward GCV MARS or full reversible-jump BMARS. Documentation should describe it as a MARS-basis/Bayesian-linear surrogate with optional model-space MCMC, not as a complete reference implementation.

## Done Criteria For The Code Port

- All public designer strings from `docs/mars_designer_configs.md` either work or are deliberately deferred in docs and tests.
- `MarsSurrogate` and `BayesianMarsSurrogate` satisfy the ENN surrogate protocol used by the local Python optimizer.
- BMARS consumes `y_var` when provided and preserves stable defaults from the TR worktree.
- Active-subspace variants call `set_low_rank_factor` on a compatible trust-region state.
- `kiss check`, `ruff check .`, and `pytest -sv tests` pass.
