# HalfCheetah MLP Surrogate Comparison

This directory contains the first HalfCheetah-v5 comparison for the MARS-family
surrogates. The question is whether the lightweight MARS/BMARS variants improve
policy search over simple non-GP baselines at the same evaluation budget.

Common protocol:

- `env_tag = "cheetah"`
- `policy_tag = "mlp-16-16"`
- `num_arms = 1`
- `num_rounds = 10000`
- `num_reps = 10`
- CPU environment runtime with trace logging enabled

Baselines:

- `random`
- `sobol`
- `turbo-enn-p` with a flat exact-neighbor index
- `turbo-enn-fit-ucb` with a flat exact-neighbor index
- `turbo-enn-varentropy-ucb` with HNSW neighbor search and bounded candidate scoring

MARS-family candidates:

- `turbo-mars-thompson`
- `turbo-bmars-thompson` with a deterministic basis
- `turbo-bmars-thompson` with MCMC basis sampling
- `turbo-mars-enn-thompson`

Run a config locally with:

```bash
./ops/experiment.py local configs/bo/gymnasium/cheetah/first_experiment/<config>.toml
```

Run a config on Modal with:

```bash
modal run ops/modal_pixi_setup.py --config configs/bo/gymnasium/cheetah/first_experiment/<config>.toml
```
