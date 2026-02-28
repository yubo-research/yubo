from __future__ import annotations

from dataclasses import dataclass

import torch

from .common import select_device


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


@dataclass
class TorchRLRuntimeConfig:
    device: str = "auto"
    collector_backend: str = "auto"
    single_env_backend: str = "auto"
    collector_workers: int | None = None

    def runtime_num_envs(self) -> int:
        return 1

    def runtime_request(self) -> TorchRLRuntimeRequest:
        return TorchRLRuntimeRequest(
            device=self.device,
            collector_backend=self.collector_backend,
            single_env_backend=self.single_env_backend,
            num_envs=int(self.runtime_num_envs()),
            collector_workers=self.collector_workers,
        )

    def resolve_runtime(
        self,
        *,
        capabilities: TorchRLRuntimeCapabilities | None = None,
    ) -> TorchRLRuntime:
        resolved_capabilities = capabilities if capabilities is not None else TorchRLRuntimeCapabilities()
        return resolve_torchrl_runtime(
            self.runtime_request(),
            capabilities=resolved_capabilities,
        )


def resolve_torchrl_runtime(
    request: TorchRLRuntimeRequest,
    *,
    capabilities: TorchRLRuntimeCapabilities = TorchRLRuntimeCapabilities(),
) -> TorchRLRuntime:
    num_envs = int(request.num_envs)
    if num_envs <= 0:
        raise ValueError(f"num_envs must be > 0, got {request.num_envs}.")

    resolved_device = select_device(request.device)
    resolved_collector_backend = _resolve_collector_backend(
        request.collector_backend,
        num_envs=num_envs,
        device=resolved_device,
        capabilities=capabilities,
    )
    resolved_single_env_backend = _resolve_single_env_backend(
        request.single_env_backend,
        num_envs=num_envs,
        collector_backend=resolved_collector_backend,
        device=resolved_device,
        capabilities=capabilities,
    )
    resolved_collector_workers = _resolve_collector_workers(
        request.collector_workers,
        num_envs=num_envs,
        collector_backend=resolved_collector_backend,
    )
    return TorchRLRuntime(
        device=resolved_device,
        collector_backend=resolved_collector_backend,
        single_env_backend=resolved_single_env_backend,
        collector_workers=resolved_collector_workers,
    )


def _normalize_collector_backend(collector_backend: str) -> str:
    normalized = str(collector_backend).strip().lower()
    valid = {"auto", "single", "multi_sync", "multi_async"}
    if normalized not in valid:
        raise ValueError("collector_backend must be one of: auto, single, multi_sync, multi_async")
    return normalized


def _normalize_single_env_backend(single_env_backend: str) -> str:
    normalized = str(single_env_backend).strip().lower()
    valid = {"auto", "serial", "parallel"}
    if normalized not in valid:
        raise ValueError("single_env_backend must be one of: auto, serial, parallel")
    return normalized


def _resolve_collector_backend(
    collector_backend: str,
    *,
    num_envs: int,
    device: torch.device,
    capabilities: TorchRLRuntimeCapabilities,
) -> str:
    normalized = _normalize_collector_backend(collector_backend)
    if normalized == "auto":
        if num_envs <= 1:
            return "single"
        if device.type == "mps" and not capabilities.allow_mps_multi_collectors:
            return "single"
        if capabilities.allow_multi_sync_collector:
            return "multi_sync"
        if capabilities.allow_multi_async_collector:
            return "multi_async"
        return "single"

    if normalized == "multi_sync" and not capabilities.allow_multi_sync_collector:
        raise ValueError("collector_backend='multi_sync' is disabled for this algorithm.")
    if normalized == "multi_async" and not capabilities.allow_multi_async_collector:
        raise ValueError("collector_backend='multi_async' is disabled for this algorithm.")
    if normalized in {"multi_sync", "multi_async"} and device.type == "mps":
        if not capabilities.allow_mps_multi_collectors:
            raise ValueError(
                "collector_backend='multi_sync'/'multi_async' is not supported with device='mps'. Use collector_backend='single' (or 'auto') on MPS."
            )
    return normalized


def _resolve_single_env_backend(
    single_env_backend: str,
    *,
    num_envs: int,
    collector_backend: str,
    device: torch.device,
    capabilities: TorchRLRuntimeCapabilities,
) -> str:
    normalized = _normalize_single_env_backend(single_env_backend)
    if collector_backend != "single":
        return "n/a"
    if normalized == "auto":
        use_parallel = num_envs > 1 and capabilities.allow_parallel_single_env and (device.type != "mps" or capabilities.allow_mps_parallel_single_env)
        if use_parallel:
            return "parallel"
        return "serial"
    if normalized == "parallel" and not capabilities.allow_parallel_single_env:
        raise ValueError("single_env_backend='parallel' is disabled for this algorithm.")
    return normalized


def _resolve_collector_workers(
    collector_workers: int | None,
    *,
    num_envs: int,
    collector_backend: str,
) -> int | None:
    if collector_backend == "single":
        return None
    resolved_workers = int(collector_workers) if collector_workers is not None else int(num_envs)
    if resolved_workers <= 0:
        raise ValueError(f"collector_workers must be > 0, got {resolved_workers}.")
    if resolved_workers != int(num_envs):
        raise ValueError(f"For multi collectors, collector_workers must equal num_envs (got collector_workers={resolved_workers}, num_envs={num_envs}).")
    return resolved_workers
