# Ackley-10d Bounded Local Sample Efficiency

This benchmark tests whether MARS-style surrogates improve local sample
efficiency over TuRBO-ENN when each optimizer is run with bounded
per-iteration surrogate work.

The target is `f:ackley-10d` with `pure-function` policy parameters. Each config
uses 10 replications, 1000 BO rounds, and 1 arm per round. The comparison is
only against TuRBO-ENN; random and Sobol baselines are intentionally omitted.
MARS and BMARS cap their training window (`trailing_obs=128`), basis size,
feature screen, and candidate count. The ENN baseline uses exact flat indexing
with fixed `k=10`.

Run all configs:

```bash
for cfg in configs/bo/synthetic/local_sample_efficiency/*.toml; do
  .pixi/envs/yubo/bin/python ops/experiment.py local "$cfg"
done
```

Plot completed runs:

```bash
.pixi/envs/yubo/bin/python -m analysis.plot_synthetic_local_sample_efficiency \
  --exp-dir runs/bo/synthetic/local_sample_efficiency/ackley10_bounded
```
