from __future__ import annotations

from dataclasses import dataclass, field, replace

from .mars_config import MarsSurrogateConfig


@dataclass(frozen=True)
class MarsENNSurrogateConfig:
    basis: MarsSurrogateConfig = field(default_factory=MarsSurrogateConfig)
    k: int = 10
    num_fit_candidates: int = 200
    num_fit_samples: int = 200
    scale_x: bool = True
    index_driver: str = "flat"
    include_noise_in_sigma: bool = False
    infer_aleatoric_variance_scale: bool = True

    def __post_init__(self) -> None:
        _check_at_least("k", self.k, 1)
        _check_at_least("num_fit_candidates", self.num_fit_candidates, 1)
        _check_at_least("num_fit_samples", self.num_fit_samples, 1)
        if self.index_driver not in {"flat", "hnsw", "hnsw_disk"}:
            raise ValueError("index_driver must be one of: flat, hnsw, hnsw_disk")

    @property
    def trailing_obs(self) -> int | None:
        return self.basis.trailing_obs

    @property
    def active_rank(self) -> int:
        return self.basis.active_rank

    @property
    def active_samples(self) -> int:
        return self.basis.active_samples

    def with_active_rank(self, active_rank: int) -> MarsENNSurrogateConfig:
        return replace(self, basis=replace(self.basis, active_rank=int(active_rank)))


def _check_at_least(name: str, value: int, minimum: int) -> None:
    if int(value) < int(minimum):
        raise ValueError(f"{name} must be >= {minimum}")
