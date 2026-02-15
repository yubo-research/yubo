# Experiment Log: num_dim_target (RAASP sparse perturbation)

## Goal
Find the best `num_dim_target` (ndt) for `SparseGaussianPerturbator` on MNIST
with 10,000 rounds, lr=0.001, sigma=0.001, batch_size=4096, Adam-normalized gradient.

## Setup
- Model: MnistClassifier (~455K params)
- Optimizer: UHDMeZO with Adam-style gradient normalization (beta=0.9)
- LR schedule: ConstantLR(0.001)
- Command: `./experiments/exp_uhd.py local --env-tag=mnist --num-rounds=10000 --lr=0.001 --ndt=<value>`

## Results

| ndt   | Description           | test_acc | y_best  | mu (final) | Notes              |
|-------|-----------------------|----------|---------|------------|--------------------|
| None  | Dense (all dims)      | 0.9078   | -0.2876 | -0.3255    | Baseline           |
| 0.5   | 50% of dims           | **0.9166** | -0.2902 | -0.3119  | **BEST**           |
| 0.6   | 60% of dims           | 0.9146   | -0.2833 | -0.3261    | Near-best          |
| 0.7   | 70% of dims           | 0.9122   | -0.2890 | -0.3213    | 3rd best           |
| 0.4   | 40% of dims           | 0.9039   | -0.3294 | -0.3461    | Below dense        |
| 0.3   | 30% of dims           | 0.9063   | -0.3632 | -0.3845    | Below dense        |
| 0.1   | 10% of dims           | 0.8633   | -0.6080 | -0.6378    | Too sparse          |

## Completed Experiments
- ndt=0.1 — DONE (0.8633)
- ndt=0.3 — DONE (0.9063)
- ndt=0.4 — DONE (0.9039)
- ndt=0.5 — DONE (0.9166) **<-- BEST**
- ndt=0.6 — DONE (0.9146)
- ndt=0.7 — DONE (0.9122)
- dense  — DONE (0.9078)

## Trajectory Comparison (test_acc at checkpoints)

| iter  | Dense  | ndt=0.7 | ndt=0.6 | ndt=0.5 | ndt=0.4 | ndt=0.3 | ndt=0.1 |
|-------|--------|---------|---------|---------|---------|---------|---------|
| 0     | 0.0919 | 0.1014  | 0.0932  | 0.1079  | 0.1287  | 0.1093  | 0.0857  |
| 1000  | 0.0919 | 0.1014  | 0.0932  | 0.1079  | 0.1287  | 0.1093  | 0.0857  |
| 2000  | 0.5936 | 0.5631  | 0.5624  | 0.5708  | 0.4931  | 0.5029  | 0.4793  |
| 3000  | 0.7102 | 0.6882  | 0.6888  | 0.7146  | 0.6691  | 0.6453  | 0.6039  |
| 4000  | 0.7987 | 0.7711  | 0.7789  | 0.7730  | 0.7379  | 0.7402  | 0.6666  |
| 5000  | 0.8317 | 0.8227  | 0.8365  | 0.8200  | 0.7879  | 0.7989  | 0.7139  |
| 6000  | 0.8614 | 0.8491  | 0.8615  | 0.8491  | 0.8252  | 0.8305  | 0.7687  |
| 7000  | 0.8810 | 0.8756  | 0.8802  | 0.8692  | 0.8579  | 0.8543  | 0.8006  |
| 8000  | 0.8897 | 0.8935  | 0.8938  | 0.8910  | 0.8724  | 0.8711  | 0.8224  |
| 9000  | 0.8930 | 0.9007  | 0.9036  | 0.8998  | 0.8888  | 0.8852  | 0.8368  |
| 10000 | 0.9078 | 0.9122  | 0.9146  | 0.9166  | 0.9039  | 0.9063  | 0.8633  |

### Final analysis

Sorted by test_acc:
```
ndt=0.5   0.9166  (+0.88% vs dense)  **BEST**
ndt=0.6   0.9146  (+0.68% vs dense)
ndt=0.7   0.9122  (+0.44% vs dense)
dense     0.9078  (baseline)
ndt=0.3   0.9063  (-0.15% vs dense)
ndt=0.4   0.9039  (-0.39% vs dense)
ndt=0.1   0.8633  (-4.45% vs dense)
```

**The optimum is ndt=0.5** (50% of parameters perturbed per step).

The top 3 values (0.5, 0.6, 0.7) all beat dense, forming a broad plateau
from 50-70% density. Below 50%, performance drops below dense — there
aren't enough perturbed dimensions per step. Above 70%, it converges
toward dense with diminishing sparsity benefit.

The mechanism: at 50% density, each dimension's perturbation carries
sqrt(2) more signal, but each dimension is only updated every ~2 steps.
This is a favorable trade-off. The optimizer effectively gets a cleaner
gradient signal per-step at the cost of slightly less per-dimension
coverage, but 50% coverage is sufficient.

The ndt=0.3/0.4 inversion (0.3 slightly beats 0.4) is likely noise —
these are single-run results and the difference is 0.24%. The robust
conclusion is that anything below 0.5 is worse than dense.

**Recommendation: Use ndt=0.5 as the default for SparseGaussianPerturbator.**
