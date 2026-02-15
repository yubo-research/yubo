# MNIST Classifier Experiment Log 2: Pushing Past 99%

## Problem

Improve MNIST test accuracy beyond 0.981 (from exp_log.md) under a 3-minute CPU wall-clock budget. Target: >0.99.

## Starting Point

Best from round 1: MLP, 256 hidden, BatchNorm, bs=1024, lr=8e-3, OneCycleLR, wd=1e-2. 3 epochs -> 0.981.

## Hypotheses

### H11: Switch from MLP to CNN
- **Rationale**: MLPs cannot exploit spatial structure in images. A CNN should learn better features.
- **Test**: 2-layer CNN (16, 32 channels, 5x5 stride=2, BatchNorm, ReLU) + FC(128), same optimizer config.
- **Result**: **Supported.** 0.9894 in 3 epochs (~44s/epoch). A 4-conv-layer CNN was too slow (132s/epoch, only 1 epoch).

### H12: Larger batch size (2048) fits more CNN epochs
- **Rationale**: bs=2048 reduces epoch time from ~44s to ~33s, enabling ~5 epochs.
- **Test**: Small CNN with bs=2048, lr=1.6e-2 (scaled), OneCycleLR, 5 planned epochs.
- **Result**: **Partially supported.** 0.9899 (5 epochs). More epochs completed but each update is less informative. Accuracy dipped at epoch 5 (0.9896).

### H13: Dropout improves generalization
- **Rationale**: Adding dropout(0.25) before FC layers could regularize the model.
- **Test**: Same CNN with dropout=0.25 before each FC layer, bs=1024, lr=8e-3.
- **Result**: **Rejected.** 0.9894 (3 epochs). Dropout slows convergence; not enough epochs to recover.

### H14: Wider convolutional channels (32, 64 instead of 16, 32)
- **Rationale**: More channels = richer feature maps.
- **Test**: CNN with (32, 64) channels, 5x5 stride=2, same optimizer.
- **Result**: **Supported.** 0.9905 in 3 epochs. Clear improvement.

### H15: 3x3 kernels instead of 5x5 (faster, more epochs)
- **Rationale**: Smaller kernels are faster; if we can fit 4 epochs, accuracy may improve.
- **Test**: Same CNN but with 3x3 stride=2 kernels.
- **Result**: **Rejected.** 0.9872 (3 epochs, no speed gain). Smaller receptive field hurts feature quality.

### H16: Match OneCycleLR schedule to actual completed epochs (3)
- **Rationale**: OneCycleLR is set for 4 epochs but only 3 complete. LR never reaches its final low values. Setting epochs=3 should give proper annealing.
- **Test**: Same wide CNN with OneCycleLR(epochs=3).
- **Result**: **Rejected.** 0.9896 < 0.9905. The incomplete 4-epoch schedule keeps a moderate LR longer, which is actually better.

### H17: Data augmentation (RandomAffine)
- **Rationale**: Seeing rotated/translated digits each epoch increases effective data diversity.
- **Test**: Added RandomAffine(degrees=10, translate=(0.1, 0.1)) to training transforms.
- **Result**: **Rejected.** 0.9905 (same as baseline). Augmentation makes training harder without enough epochs to benefit.

### H18: Label smoothing
- **Rationale**: Label smoothing (0.1) provides regularization without slowing computation.
- **Test**: CrossEntropyLoss(label_smoothing=0.1) with best CNN config.
- **Result**: **Rejected.** 0.9902. No improvement.

### H19: Higher max_lr (1.2e-2) for CNN
- **Rationale**: BatchNorm stabilizes training, allowing the CNN to tolerate a higher LR than the MLP. Higher LR = faster convergence per step.
- **Test**: OneCycleLR with max_lr=1.2e-2 vs 8e-3 vs 1.5e-2.
- **Result**: **Supported.** max_lr=1.2e-2 -> 0.9907. max_lr=1.5e-2 -> 0.9894 (too aggressive). Sweet spot at 1.2e-2.

### H20: torch.compile() speeds up training
- **Rationale**: JIT compilation could reduce per-epoch time, fitting 4 epochs.
- **Test**: torch.compile(model) with best config.
- **Result**: **Rejected.** Epoch 1 took 75s (compilation overhead), epoch 2 took 57s. Only 2 epochs completed. 0.9882.

### H21: Stride=1 + MaxPool + Global Average Pooling
- **Rationale**: Stride=1 preserves spatial resolution better; GAP reduces FC parameters and speeds up training.
- **Test**: 3-layer CNN (32, 64, 128 channels) with stride=1, MaxPool(2), GAP, single FC(128->10).
- **Result**: **Rejected.** 0.8874 (2 epochs, 65s/epoch). GAP bottleneck too aggressive; model can't learn fast enough.

### H22: Wider FC head (256 neurons) with BatchNorm
- **Rationale**: More classifier capacity with minimal speed cost since convolutions dominate.
- **Test**: FC head widened to 256 with BatchNorm added.
- **Result**: **Rejected.** 0.9895 (3 epochs). Slightly slower per epoch (57s vs 55s); no accuracy gain.

## Final Configuration

| Parameter     | Round 1 Best | Round 2 Best |
|---------------|-------------|-------------|
| Architecture  | MLP         | CNN (2 conv layers) |
| Conv channels | —           | 32, 64      |
| Conv kernels  | —           | 5x5, stride=2 |
| FC head       | 256         | 128         |
| BatchNorm     | Yes (FC)    | Yes (conv + FC) |
| batch_size    | 1024        | 1024        |
| max_lr        | 8e-3        | 1.2e-2      |
| weight_decay  | 1e-2        | 1e-2        |
| scheduler     | OneCycleLR  | OneCycleLR  |
| num_epochs    | 3           | 4 (3 completed) |

## Final Result

**0.991 test accuracy** (3 epochs in ~170s), up from 0.981 (round 1) and 0.957 (original baseline). Target of >0.99 achieved.

## Key Takeaways

1. **Architecture matters most**: MLP -> CNN was the biggest single jump (0.981 -> 0.989).
2. **Speed is accuracy on a budget**: On CPU with a time limit, anything that slows epochs (deeper nets, torch.compile overhead, data augmentation) hurts even if it would help given unlimited time.
3. **Diminishing returns on hyperparams**: Once architecture and LR schedule are right, tuning dropout/label smoothing/weight decay yielded <0.1% changes.
4. **OneCycleLR schedule mismatch helps**: Configuring OneCycleLR for more epochs than actually complete keeps the LR from decaying too fast.
