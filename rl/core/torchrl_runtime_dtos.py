from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TorchRLRuntimeCapabilities:
    allow_multi_sync_collector: bool = True
    allow_multi_async_collector: bool = True
    allow_mps_multi_collectors: bool = False
    allow_parallel_single_env: bool = True
    allow_mps_parallel_single_env: bool = True


@dataclass(frozen=True)
class TorchRLRuntimeRequest:
    device: str = "auto"
    collector_backend: str = "auto"
    single_env_backend: str = "auto"
    num_envs: int = 1
    collector_workers: int | None = None


@dataclass(frozen=True)
class TorchRLRuntime:
    device: torch.device
    collector_backend: str
    single_env_backend: str
    collector_workers: int | None
