from __future__ import annotations

from problems.isaaclab_env_adapters import is_isaaclab_env_tag

from .torchrl_runtime_dtos import TorchRLRuntimeRequest


def make_torchrl_runtime_request(
    *,
    env_tag: str,
    device: str,
    collector_backend: str,
    single_env_backend: str,
    num_envs: int,
    collector_workers: int | None,
) -> TorchRLRuntimeRequest:
    # Warp and IsaacLab handle their own GPU vectorization, so we use the 'single' collector
    if is_isaaclab_env_tag(env_tag) or str(env_tag).startswith("warp:"):
        collector_backend = "single"
        single_env_backend = "serial"
        collector_workers = 1

    return TorchRLRuntimeRequest(
        device=device,
        collector_backend=str(collector_backend),
        single_env_backend=str(single_env_backend),
        num_envs=int(num_envs),
        collector_workers=collector_workers,
    )
