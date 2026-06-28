from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ENNVarentropySurrogateConfig:
    k: int = 10
    index_driver: str = "flat"
    scale_x: bool = False
    varentropy_scale: float = 0.5
    variance_eps: float = 1e-12
    normalize_varentropy: bool = True
    include_noise_in_sigma: bool = False

    def __post_init__(self) -> None:
        _check_at_least("k", self.k, 1)
        if self.index_driver not in {"flat", "hnsw", "hnsw_disk"}:
            raise ValueError("index_driver must be one of: flat, hnsw, hnsw_disk")
        if self.varentropy_scale < 0.0:
            raise ValueError("varentropy_scale must be non-negative")
        if self.variance_eps <= 0.0:
            raise ValueError("variance_eps must be positive")


def _check_at_least(name: str, value: int, minimum: int) -> None:
    if int(value) < int(minimum):
        raise ValueError(f"{name} must be >= {minimum}")


__all__ = ["ENNVarentropySurrogateConfig"]
