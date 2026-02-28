"""Configuration sub-types for UHD experiments (EarlyReject, BE, ENN)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EarlyRejectConfig:
    tau: float | None
    mode: str | None
    ema_beta: float | None
    warmup_pos: int | None
    quantile: float | None
    window: int | None


@dataclass(frozen=True)
class BEConfig:
    num_probes: int
    num_candidates: int
    warmup: int
    fit_interval: int
    enn_k: int
    sigma_range: tuple[float, float] | None


@dataclass(frozen=True)
class ENNConfig:
    minus_impute: bool
    d: int
    s: int
    jl_seed: int
    k: int
    fit_interval: int
    warmup_real_obs: int
    refresh_interval: int
    se_threshold: float
    target: str
    num_candidates: int
    select_interval: int
    embedder: str
    gather_t: int
