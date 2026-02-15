# Experiment Log: num_module_target (submodule-level RAASP)

## Goal
Find the best `num_module_target` (nmt) for `SubmodulePerturbator` on MNIST
with 10,000 rounds, lr=0.001, sigma=0.001, batch_size=4096, Adam-normalized gradient.

## Setup
- Model: MnistClassifier (~455K params, 6 leaf modules with params)
- Leaf modules: Conv2d(1,32)[832], BN(32)[64], Conv2d(32,64)[51264],
  BN(64)[128], Linear(3136,128)[401536], Linear(128,10)[1290]
- Note: Linear(3136,128) holds 88% of all parameters.
- Optimizer: UHDMeZO with Adam-style gradient normalization (beta=0.9)
- LR schedule: ConstantLR(0.001)
- Command: `./experiments/exp_uhd.py local --env-tag=mnist --num-rounds=10000 --perturb=mod:<value>`

## Baselines (from ndt experiments)
- Dense: 0.9078
- dim:0.5 (best dim-level): 0.9166

## Results

| nmt   | Modules/step (avg) | test_acc | y_best  | mu (final) | Notes              |
|-------|--------------------|----------|---------|------------|--------------------|
| 1     | 1 of 6             | 0.8697   | -0.5043 | -0.5340    | Too sparse         |
| 2     | 2 of 6             | 0.8985   | -0.3682 | -0.4045    | Below dense        |
| 3     | 3 of 6             | 0.9017   | -0.3287 | -0.3428    | Below dense        |
| 4     | 4 of 6             | 0.9024   | -0.3199 | -0.3637    | Below dense        |
| 5     | 5 of 6             | 0.9108   | -0.2816 | -0.3192    | Above dense        |

## Completed Experiments
- nmt=1 — DONE (0.8697)
- nmt=2 — DONE (0.8985)
- nmt=3 — DONE (0.9017)
- nmt=4 — DONE (0.9024)
- nmt=5 — DONE (0.9108) **<-- BEST**

## Trajectory Comparison (test_acc at checkpoints)

| iter  | Dense  | dim:0.5 | mod:5  | mod:4  | mod:3  | mod:2  | mod:1  |
|-------|--------|---------|--------|--------|--------|--------|--------|
| 0     | 0.0919 | 0.1079  | 0.1134 | 0.1131 | 0.1092 | 0.0988 | 0.0893 |
| 1000  | 0.0919 | 0.1079  | 0.1134 | 0.1131 | 0.1092 | 0.0988 | 0.0893 |
| 2000  | 0.5936 | 0.5708  | 0.6033 | 0.5211 | 0.5011 | 0.5266 | 0.3898 |
| 3000  | 0.7102 | 0.7146  | 0.7371 | 0.6626 | 0.6243 | 0.6425 | 0.5626 |
| 4000  | 0.7987 | 0.7730  | 0.7996 | 0.7591 | 0.7287 | 0.7217 | 0.6525 |
| 5000  | 0.8317 | 0.8200  | 0.8406 | 0.8051 | 0.7953 | 0.7627 | 0.7181 |
| 6000  | 0.8614 | 0.8491  | 0.8673 | 0.8322 | 0.8237 | 0.8056 | 0.7612 |
| 7000  | 0.8810 | 0.8692  | 0.8893 | 0.8681 | 0.8502 | 0.8318 | 0.7870 |
| 8000  | 0.8897 | 0.8910  | 0.8977 | 0.8758 | 0.8711 | 0.8567 | 0.8157 |
| 9000  | 0.8930 | 0.8998  | 0.9065 | 0.8837 | 0.8828 | 0.8784 | 0.8379 |
| 10000 | 0.9078 | 0.9166  | 0.9108 | 0.9024 | 0.9017 | 0.8985 | 0.8697 |

### Final analysis

Sorted by test_acc:
```
dim:0.5   0.9166  (+0.88% vs dense)  **BEST overall**
mod:5     0.9108  (+0.30% vs dense)  **BEST submodule**
dense     0.9078  (baseline)
mod:4     0.9024  (-0.54% vs dense)
mod:3     0.9017  (-0.61% vs dense)
mod:2     0.8985  (-0.93% vs dense)
mod:1     0.8697  (-3.81% vs dense)
```

**The optimal num_module_target is 5** (5 of 6 modules per step).

However, submodule-level RAASP is clearly inferior to dimension-level
RAASP on this model. The best submodule result (mod:5 = 0.9108) is
below the best dimension result (dim:0.5 = 0.9166) and only barely
beats dense (0.9078).

### Why submodule RAASP underperforms

The core issue is **extreme parameter-count imbalance**. The 6 leaf
modules have wildly different sizes:

```
Linear(3136,128)  401,536 params  (88.2%)
Conv2d(32,64)      51,264 params  (11.3%)
Linear(128,10)      1,290 params   (0.3%)
Conv2d(1,32)          832 params   (0.2%)
BatchNorm2d(64)       128 params   (0.03%)
BatchNorm2d(32)        64 params   (0.01%)
```

When a module is excluded, ALL of its parameters are excluded. Excluding
the big Linear layer (88% of params) is like setting ndt=0.12 for that
step — far below the optimal 0.5. In contrast, dimension-level RAASP
always perturbs 50% of every layer's parameters uniformly.

The monotonic trend (more modules = better) confirms this: the optimizer
simply needs most modules perturbed most of the time. With only 6
modules, there's not enough granularity to find a "sweet spot" like
dim:0.5 — the coarsest useful sparsity is 5/6 ≈ 83%.

**Recommendation: Prefer dim:0.5 over mod:<any> for this architecture.
Submodule-level RAASP may be more useful on architectures with many
similarly-sized modules (e.g., transformers with many attention heads).**
