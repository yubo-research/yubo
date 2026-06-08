from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunFields:
    env_tag: str
    num_rounds: int
    policy_tag: str | None
    lr: float
    sigma: float
    ndt: int | None
    nmt: int | None
    problem_seed: int | None
    noise_seed_0: int | None
    log_interval: int
    accuracy_interval: int
    target_accuracy: float | None


@dataclass(frozen=True)
class EarlyRejectFields:
    tau: float | None
    mode: str | None
    ema_beta: float | None
    warmup_pos: int | None
    quantile: float | None
    window: int | None
