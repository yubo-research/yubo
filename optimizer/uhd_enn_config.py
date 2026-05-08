from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ENNImputerConfig:
    d: int = 100
    s: int = 4
    jl_seed: int = 123
    k: int = 25
    fit_interval: int = 50  # refit after this many new real observations
    warmup_real_obs: int = 200  # don't impute until we have this many real points
    refresh_interval: int = 50  # force a real negative eval every N negative phases
    se_threshold: float = 0.25  # if predicted se > threshold, do real eval instead
    # What to model with ENN:
    # - "mu_minus": predict mu(x-) directly (original prototype).
    # - "delta": predict Δμ = mu(x+) - mu(x-) and convert back to mu(x-).
    # - "mu_plus": predict mu(x+) from direction embeddings (for seed selection only).
    target: str = "mu_minus"  # "mu_minus" | "delta" | "mu_plus"
    # If >1 (and target="delta"), we can do seed filtering by scoring multiple
    # candidate direction embeddings and choosing max UCB.
    num_candidates: int = 1
    select_interval: int = 1
    # Embedding backend:
    # - "direction": use SparseJL direction embeddings from k-sparse noise (fast, but requires nz noise).
    # - "gather": sparse-by-row gather projection of current module params (no O(D) buffers).
    embedder: str = "direction"  # "direction" | "gather"
    gather_t: int = 64
    # Calibration gate: only impute if recent prediction error is low.
    err_ema_beta: float = 0.95
    max_abs_err_ema: float = 0.25
    min_calib_points: int = 10


def _compute_z(z_base: torch.Tensor, delta_z: torch.Tensor, sign: float) -> np.ndarray:
    return (z_base + sign * delta_z).double().numpy()
