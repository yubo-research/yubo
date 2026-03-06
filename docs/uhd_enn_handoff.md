# UHD + ENN + SparseJL Integration Handoff

Goal: integrate SparseJL embeddings + ENN surrogate predictions into the UHDMeZO loop to reduce expensive real evaluations and/or choose better perturbation directions.

This is a **handoff doc** for another agent to start coding and running experiments. It includes:
- current UHDMeZO mechanics and where to hook in,
- SparseJL / Delta-JL APIs in this repo,
- how ENN is used in this codebase (fit + posterior),
- concrete integration variants,
- suggested experiments and success metrics.

## Current state of the codebase (relevant pointers)

- **UHDMeZO**: `optimizer/uhd_mezo.py`
  - Two-phase antithetic sampling (`ask()`/`tell()` twice per effective update).
  - Computes scalar projected gradient from \(\mu_+,\mu_-\) and applies an Adam-style scalar normalization.
- **UHDLoop** (orchestrates evaluation and prints progress): `optimizer/uhd_loop.py`
- **MNIST UHD CLI** (TOML-only): `ops/exp_uhd.py`
  - Example config: `configs/mnist_uhd.toml`
- **SparseJL transform** (Torch): `sampling/sparse_jl_t.py`
  - `block_sparse_jl_transform_t(x, d, s=4, seed=42) -> y`
  - `block_sparse_jl_transform_module(module, d, s=4, seed=42) -> y`
- **Delta SparseJL**: `sampling/delta_sparse_jl_t.py`
  - `DeltaSparseJL_T(num_dim_ambient=D, num_dim_embedding=d, s=4, seed=42, incremental=...)`
  - supports `initialize(x0)` and `transform(d_x_sparse)` which returns either `T(x0 + d_x)` or `T(x0) + T(d_x)` (incremental path).

## UHDMeZO recap (exact algorithm implemented here)

UHDMeZO is gradient ascent with antithetic sampling (zeroth-order). Each *effective* update uses two evaluations:

- Evaluate at \(x^+ = x + \sigma \varepsilon\) → observe \(\mu_+\)
- Evaluate at \(x^- = x - \sigma \varepsilon\) → observe \(\mu_-\)
- Estimate projected gradient:

\[
g_{\text{proj}} = \frac{\mu_+ - \mu_-}{2\sigma}.
\]

- Adam-style scalar normalization:
  - EMA of squared projected grad: \(v \leftarrow \beta v + (1-\beta) g_{\text{proj}}^2\)
  - RMS: \(\sqrt{v} + 10^{-8}\)
  - Step scale: `step_scale = lr * g_proj / rms`
- Apply update in the same direction: \(x \leftarrow x + \text{step_scale}\cdot \varepsilon\)

Important: **no D-sized noise vector is stored**; perturbators regenerate noise by seed.

Implementation details to preserve:
- `ask()` toggles positive/negative phase; `tell()` does unperturb and (in negative phase) applies update.
- `eval_seed` increments once per *full* (positive+negative) cycle.

## ENN usage model (what you need to assume)

In this codebase, ENN is treated as providing:
- a posterior mean \(\mu(x)\)
- a calibrated epistemic uncertainty `se(x)`

Given new observations, you call `enn_fit()`; for candidate scoring you only call `posterior()`.

Canonical pattern:

```python
from third_party.enn import EpistemicNearestNeighbors, enn_fit
from third_party.enn.enn.enn_params import PosteriorFlags

# Fit/update when new data arrive:
enn_model = EpistemicNearestNeighbors(X_train, y_train[:, None])
enn_params = enn_fit(enn_model, k=k, num_fit_candidates=..., num_fit_samples=..., rng=rng)

# Predict and rank candidates without refitting:
flags = PosteriorFlags(observation_noise=False)
post = enn_model.posterior(X_cand, params=enn_params, flags=flags)
mu = post.mu
se = post.se
ucb = mu + se  # Per project assumption: no multiplier needed
```

Notes:
- Many call sites use `y[:, None]` (2D); keep shapes consistent.
- If you standardize `y`, unstandardize `mu` and `se` consistently.

## Why SparseJL makes sense here

SparseJL is a linear map \(T\). For UHD perturbations \(x^\pm = x \pm \sigma \varepsilon\):

\[
T(x^\pm) = T(x) \pm \sigma T(\varepsilon)
\]

and
\[
\Delta z := T(x^+) - T(x^-) = 2\sigma T(\varepsilon).
\]

So you can avoid embedding full \(x^\pm\) repeatedly if you maintain:
- base embedding \(z=T(x)\)
- per-candidate direction embedding \(e=T(\varepsilon)\)

Then:
- \(z^+=z+\sigma e\)
- \(z^-=z-\sigma e\)

This is the primary efficiency win.

## Integration variants (implementation sketches)

### Variant A: ENN as a cheap filter to pick which direction to evaluate (recommended first)

Use ENN + UCB to rank many candidate antithetic directions, then evaluate only the best few with the real environment.

**Mechanics**
- Maintain the current embedded state \(z=T(x)\).
- For each UHD step, sample K candidate seeds \(\{s_i\}\).
  - For each seed, compute `e_i = T(ε(seed=s_i))` in \(R^d\).
  - Compute `z_plus_i = z + sigma * e_i`, `z_minus_i = z - sigma * e_i`.
  - Use ENN posterior to get `(mu_plus_i, se_plus_i)` and `(mu_minus_i, se_minus_i)`.
  - Score each seed using UCB-style surrogate:
    - `ucb_plus = mu_plus + se_plus`
    - `ucb_minus = mu_minus + se_minus`
    - a direction-score could be:
      - maximize predicted improvement: `max(ucb_plus, ucb_minus)`, or
      - maximize predicted |g_proj|: `abs((mu_plus - mu_minus) / (2*sigma))`, or
      - combine: `abs(mu_plus - mu_minus) + (se_plus + se_minus)` (heuristic).
- Choose the top-1 (or top-m) seeds.
- Run the **real** UHDMeZO antithetic evaluation on that chosen seed only (preserving real objective as truth).

**Where to hook**
- Most naturally inside `UHDMeZO.ask()` (choose seed) and/or in a wrapper around `UHDLoop.run()` that chooses which seed to pass to `perturbator.perturb(seed, ...)`.
- The cleanest code change is likely a new “seed chooser” strategy class used by `UHDMeZO` during the positive phase.

**What data to log**
- chosen seed, K, top scores, predicted `mu±/se±`, realized `mu±/se±`, realized `g_proj`.

**Hypothesis**
- Predictions: faster rise of `y_best` / test accuracy for fixed wall-clock.
- Test: A/B compare baseline vs filter (same evaluation budget), multiple random seeds.

### Variant B: Replace real \(\mu_\pm\) with ENN-predicted \(\hat\mu_\pm\) (aggressive)

Instead of evaluating the environment at both antithetic points every time, you sometimes (or always) use ENN predictions as \(\mu_\pm\).

This is closer to “surrogate gradients”. It is riskier: optimization can drift if surrogate is wrong.

Safer sub-variants:
- **B1**: surrogate-only for *negative phase selection*, still do occasional real corrections.
- **B2**: mixture policy: with probability p do real eval, else surrogate eval; always keep training data updated with real evals.

### Variant C: Model \(g_{\text{proj}}\) directly

Target is \(g_{\text{proj}} = (\mu_+ - \mu_-)/(2\sigma)\).

Inputs in embedding space:
- **C1**: `ENN([z_plus, z_minus])` in \(R^{2d}\)
- **C2**: `ENN([z, delta_z])` where `delta_z = z_plus - z_minus` in \(R^{2d}\)

Avoid “delta-only” unless you are intentionally assuming local stationarity; include base location `z` if you want robustness.

### Variant D: Use DeltaSparseJL_T for faster embedding updates

If you can represent parameter changes as a sparse delta \(d_x\), you can update embeddings as:
- `y0 = T(x0)` and `T(x0 + d_x) = y0 + T(d_x)` (incremental path).

However, UHD perturbations are typically **dense Gaussian noise** at the parameter level, so deltas are not sparse unless you:
- use a sparse perturbator (already present: `SparseGaussianPerturbator` in `optimizer/uhd_loop.py`), or
- define a sparse *proposal set* for perturbations (e.g. only perturb a subset).

So DeltaSparseJL_T is most immediately useful when:
- perturbations are sparse (dim-targeted) and you can form `d_x` as a sparse vector of changed coordinates.

## Practical engineering plan (what to code first)

### Step 0: choose d, s, seed conventions
- pick embedding dim \(d\) (e.g. 100 or 256) and sparsity \(s\) (default 4).
- fix a global JL seed (separate from UHD eval seeds) so the embedding is consistent over time.

### Step 1: implement “embed current module params” and “embed direction”
- Use `block_sparse_jl_transform_module(module, d, s, seed)` to compute `z = T(x)` without flattening.
- For directions, you need `e = T(ε(seed))`. Two routes:
  - If you can generate a dense `ε` tensor deterministically from seed: call `block_sparse_jl_transform_t(eps, d, s, jl_seed)`.
  - Better: implement a `JLNoiseEmbedding` that mirrors your perturbator’s noise generation so you do **not** build a D-vector.
    - This is the most important missing piece for performance.

### Step 2: build an ENN dataset in embedding space
Store observations:
- input features: `z` (or `z_plus/z_minus`, depending on variant)
- target: observed `mu` (and optionally observed noise model)

Fit/update:
- when new real obs arrive → append to dataset → call `enn_fit()` → cache `enn_params`.

### Step 3: implement candidate scoring + selection (Variant A)
- Sample K candidate seeds.
- For each seed, compute `e_i`, derive `z±`.
- Batch call `enn.posterior()` on all candidates (vectorized).
- Choose best by `UCB = mu + se` (or an agreed score).
- Use chosen seed in actual UHD evaluation.

### Step 4: experiments

Minimum experiments:
- **A0** Baseline UHD (no ENN, no JL).
- **A1** ENN filter with K in {4, 16, 64}.
- **A2** ENN filter + sparse perturbator vs dense perturbator.
- **A3** Compare direction scoring rules:
  - maximize `max(UCB_plus, UCB_minus)`
  - maximize `abs(mu_plus - mu_minus)` (no UCB)
  - maximize `abs(mu_plus - mu_minus) + (se_plus + se_minus)`

Metrics:
- `y_best` vs wall-clock and vs eval-count
- test accuracy (MNIST path prints occasionally)
- stability (variance across random seeds)

## Runtime / CLI notes for MNIST UHD

Current MNIST UHD runner is TOML-only:

```bash
./ops/exp_uhd.py local configs/mnist_uhd.toml
```

Config format (`configs/mnist_uhd.toml`):
- `[uhd] env_tag, num_rounds, lr, perturb`

## Gotchas / constraints

- UHDMeZO uses **two evaluations per update**; be explicit whether “budget” counts evaluations or updates.
- Be careful with random seeds:
  - UHD eval seed drives environment batches/noise pairing for antithetic.
  - JL seed must be fixed for a stable embedding.
  - Candidate direction seeds must be replayable so you can apply the chosen perturbation exactly.
- Embedding cost matters:
  - `block_sparse_jl_transform_module` is O(D*s) but memory-friendly.
  - The real win comes from embedding directions without materializing D-sized vectors.
- If using MNIST:
  - accuracy is expensive (full test set); current loop prints acc every 1000 iters.

## Suggested file/module layout for implementation

Keep integration modular so it’s easy to A/B:
- `optimizer/uhd_seed_chooser.py` (new): strategies for choosing seeds each step.
- `optimizer/jl_embedder.py` (new): parameter embedding + noise embedding utilities.
- `optimizer/enn_surrogate.py` (new): wraps ENN dataset, `enn_fit`, and `posterior` batch calls.
- small TOML config extension (optional later): `d`, `s`, `K`, strategy name.

## Definition of done (for initial integration)

- Variant A implemented behind a flag/strategy, default remains baseline behavior.
- Logs show: chosen seed, predicted scores, realized outcomes.
- Reproducible runs on MNIST that demonstrate either:
  - improved `y_best` at equal evaluation budget, or
  - similar `y_best` with fewer real evals.

## New: ENN integration ideas to reduce wall-time-to-90% (not just eval-count per step)

After basic throughput optimizations, MNIST wall-clock is dominated by the forward+loss evaluation itself. To reduce *wall-time-to-90%*, ENN should ideally be used to **avoid some real objective evaluations**, not just to rank directions.

### Trick: get both + and − “candidates” from one direction embedding
Because SparseJL is linear, if you maintain `z = T(x)` and compute one direction embedding `e = T(ε(seed))`, then:
- `z_plus  = z + sigma * e`
- `z_minus = z - sigma * e`

So you can form both surrogate inputs with a cheap reflection in embedding space (no need to embed `x±` separately). You can also batch all `z_plus`/`z_minus` into one `posterior()` call.

### Trick: keep `z = T(x)` up to date without re-embedding parameters
If the accepted update is `x ← x + step_scale * ε(seed)` (UHDMeZO), linearity gives an exact embedding update:
- `z ← z + step_scale * e`

This avoids repeated calls to `block_sparse_jl_transform_module(...)` after initialization, as long as you already computed `e` for the chosen seed.

### Option 1 (recommended first): “Surrogate negative phase” (reduce evals/update toward ~1)
Baseline UHD uses two real evaluations per effective update (`x+` and `x-`). A practical eval-saving variant:
- Always do **real** eval at `x+` to stay anchored to truth.
- Use ENN to predict `μ̂(x-)` (and uncertainty) and **skip** the real `x-` evaluation when the surrogate is confident.
- Gate the skip decision using surrogate `se` (or similar), e.g.:
  - if `se(z_minus) < τ` → use surrogate for `mu_minus`
  - else → pay for real `mu_minus`

This can reduce wall-clock directly if the skip rate is high once ENN is warmed up.

### Option 2: Early rejection / accept-reject filtering (biased but can save ~50%)
Evaluate real `x+` first; if it looks unpromising, skip `x-` and resample a new seed. You can use:
- realized `mu_plus`, and/or
- surrogate `UCB(z_minus)` to decide whether `x-` is worth paying for.

This can save evaluations, but introduces bias; treat as an empirical speed hack.

### Option 3: ENN seed filtering (Variant A) to reduce iterations-to-target
Sample `K` candidate seeds, embed directions, use ENN to score, then run the real antithetic pair only for the best seed. This does *not* reduce evaluations per selected seed, but can reduce **iterations-to-90%** enough to win overall.

### Critical engineering requirement: direction embedding without O(D) allocations
All variants depend on computing `e = T(ε(seed))` cheaply.
- Naively generating full parameter-shaped noise/masks is O(D) temporary memory and can dominate runtime.
- Preferred: a `JLNoiseEmbedding` that mirrors the perturbator’s RNG and computes `T(ε)` in a streaming/chunked way (or nnz-proportional for sparse perturbations), without materializing dense D-sized tensors.

### Minimal falsifiable experiment plan (MNIST)
A/B compare against baseline, measure:
- **time-to-0.90 test accuracy** (primary)
- number of **real** objective evals used
- skip rate of `x-` evals (for Option 1/2)
- stability across RNG seeds

Start with Option 1 using a small grid of thresholds `τ` and log: predicted `(μ̂-, se-)`, realized `μ-` when paid, and downstream `g_proj` behavior.
