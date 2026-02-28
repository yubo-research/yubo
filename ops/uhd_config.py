"""Configuration types for UHD experiments."""

from dataclasses import dataclass

from ops.uhd_config_types import BEConfig, EarlyRejectConfig, ENNConfig


@dataclass(frozen=True)
class UHDConfig:
    env_tag: str
    num_rounds: int
    problem_seed: int | None
    noise_seed_0: int | None
    lr: float
    num_dim_target: float | None
    num_module_target: float | None
    log_interval: int
    accuracy_interval: int
    target_accuracy: float | None
    optimizer: str
    batch_size: int
    early_reject: EarlyRejectConfig
    be: BEConfig
    enn: ENNConfig
    bszo_k: int
    bszo_epsilon: float
    bszo_sigma_p_sq: float
    bszo_sigma_e_sq: float
    bszo_alpha: float


__all__ = ["BEConfig", "EarlyRejectConfig", "ENNConfig", "UHDConfig"]
