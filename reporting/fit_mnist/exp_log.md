# MNIST Classifier Experiment Log

## Problem

Maximize test-set accuracy of an MNIST classifier trained with AdamW under a 3-minute wall-clock timeout.

## Baseline

- Architecture: MLP, 2 hidden layers, 128 units, ReLU
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-2
- Batch size: 128
- Result: **0.957 test accuracy**, 1 epoch completed (timed out during epoch 2)

Bottleneck: ~300ms per batch, 469 batches/epoch = ~148s/epoch. Only 1 epoch fits in 180s.

## Hypotheses

### H1: Data loading is the bottleneck (num_workers=0)
- **Rejected.** workers=4 reduced per-batch time by only ~6% (307ms -> 289ms). Compute dominates.

### H2: Larger batch size reduces batches/epoch, fitting more epochs in the budget
- **Supported.** Epoch times: bs=128: 148s, bs=512: 74s, bs=1024: 49s, bs=2048: 32s.
- bs=1024 (3 epochs): 0.971 test accuracy.

### H3: LR should scale with batch size (linear scaling rule)
- **Supported.** bs=1024 with lr=8e-3 (8x baseline): 0.959 in 60s (2 epochs). Best among tested LRs at this batch size.

### H4: BatchNorm speeds convergence
- **Supported.** Same config + BatchNorm: 0.973 (3 epochs in 180s).

### H5: Wider network (256 hidden units) improves capacity
- **Supported.** 256 hidden + BN: 0.976 (3 epochs).

### H6: Even wider (512 hidden units) helps further
- **Weakly supported.** 512 hidden + BN: 0.977 (2 epochs). Marginal gain, lost an epoch due to slower compute. 256 is the better trade-off.

### H7: Deeper network (3 hidden layers) captures more complex features
- **Rejected.** 3 layers x 256 + BN: 0.974 (2 epochs). Slower and worse than 2 layers.

### H8: OneCycleLR schedule improves convergence
- **Strongly supported.** 256 hidden + BN + OneCycleLR(max_lr=8e-3): **0.981** (3 epochs). Biggest single improvement.

### H9: Lower weight decay reduces over-regularization
- **Rejected.** wd=1e-4: 0.979, wd=1e-3: 0.979, wd=1e-2: 0.981. Default was already optimal.

### H10: Higher max_lr (2e-2) with OneCycleLR extracts more per epoch
- **Rejected.** max_lr=2e-2: 0.974 (2 epochs). Too aggressive, lost an epoch.

## Final Configuration

| Parameter    | Baseline | Final   |
|--------------|----------|---------|
| hidden_size  | 128      | 256     |
| batch_norm   | No       | Yes     |
| batch_size   | 128      | 1024    |
| lr           | 1e-3     | 8e-3    |
| weight_decay | 1e-2     | 1e-2    |
| scheduler    | None     | OneCycleLR |
| num_epochs   | 5        | 3       |

## Final Result

**0.981 test accuracy** (3 epochs completed in ~170s), up from 0.957.
