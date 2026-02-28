import pytest
import torch

from rl.torchrl.common.common import select_device
from rl.torchrl.common.runtime import (
    TorchRLRuntime,
    TorchRLRuntimeCapabilities,
    TorchRLRuntimeConfig,
    TorchRLRuntimeRequest,
    resolve_torchrl_runtime,
)


def test_select_device_auto_prefers_mps_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr("rl.torchrl.common.common.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("rl.torchrl.common.common._mps_is_available", lambda: True)
    assert select_device("auto").type == "mps"


def test_select_device_cpu_fallback(monkeypatch):
    monkeypatch.setattr("rl.torchrl.common.common.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("rl.torchrl.common.common._mps_is_available", lambda: False)
    assert select_device("auto").type == "cpu"


def test_select_device_explicit_cuda_requires_cuda(monkeypatch):
    monkeypatch.setattr("rl.torchrl.common.common.torch.cuda.is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is not available"):
        select_device("cuda")


def test_runtime_auto_uses_single_for_single_env():
    runtime = resolve_torchrl_runtime(TorchRLRuntimeRequest(num_envs=1))
    assert runtime.collector_backend == "single"
    assert runtime.single_env_backend == "serial"
    assert runtime.collector_workers is None


def test_runtime_auto_uses_multi_sync_for_multi_env_cpu():
    runtime = resolve_torchrl_runtime(
        TorchRLRuntimeRequest(
            num_envs=8,
            collector_backend="auto",
            device="cpu",
        )
    )
    assert runtime.collector_backend == "multi_sync"
    assert runtime.single_env_backend == "n/a"
    assert runtime.collector_workers == 8


def test_runtime_auto_falls_back_to_single_on_mps(monkeypatch):
    # Simulate MPS-capable host regardless of actual test machine.
    monkeypatch.setattr("rl.torchrl.common.common._mps_is_available", lambda: True)
    monkeypatch.setattr("rl.torchrl.common.common.torch.cuda.is_available", lambda: False)
    runtime = resolve_torchrl_runtime(
        TorchRLRuntimeRequest(
            num_envs=8,
            collector_backend="auto",
            single_env_backend="auto",
            device="mps",
        )
    )
    assert runtime.collector_backend == "single"
    assert runtime.single_env_backend == "parallel"
    assert runtime.collector_workers is None


def test_runtime_multi_async_rejects_mps_by_default(monkeypatch):
    monkeypatch.setattr("rl.torchrl.common.common._mps_is_available", lambda: True)
    monkeypatch.setattr("rl.torchrl.common.common.torch.cuda.is_available", lambda: False)
    with pytest.raises(ValueError, match="not supported with device='mps'"):
        resolve_torchrl_runtime(
            TorchRLRuntimeRequest(
                num_envs=8,
                collector_backend="multi_async",
                device="mps",
            )
        )


def test_runtime_mps_multi_async_allowed_when_capability_enabled(monkeypatch):
    monkeypatch.setattr("rl.torchrl.common.common._mps_is_available", lambda: True)
    monkeypatch.setattr("rl.torchrl.common.common.torch.cuda.is_available", lambda: False)
    runtime = resolve_torchrl_runtime(
        TorchRLRuntimeRequest(
            num_envs=8,
            collector_backend="multi_async",
            device="mps",
        ),
        capabilities=TorchRLRuntimeCapabilities(allow_mps_multi_collectors=True),
    )
    assert runtime.collector_backend == "multi_async"
    assert runtime.collector_workers == 8


def test_runtime_explicit_mps_requires_mps_availability(monkeypatch):
    monkeypatch.setattr("rl.torchrl.common.common._mps_is_available", lambda: False)
    with pytest.raises(ValueError, match="MPS is not available"):
        resolve_torchrl_runtime(
            TorchRLRuntimeRequest(
                num_envs=1,
                collector_backend="single",
                device="mps",
            )
        )


def test_runtime_multi_workers_must_match_num_envs():
    with pytest.raises(ValueError, match="collector_workers must equal num_envs"):
        resolve_torchrl_runtime(
            TorchRLRuntimeRequest(
                num_envs=8,
                collector_backend="multi_sync",
                device="cpu",
                collector_workers=4,
            )
        )


def test_runtime_config_helper_builds_request_and_resolves():
    class _DummyConfig(TorchRLRuntimeConfig):
        def runtime_num_envs(self) -> int:
            return 8

    config = _DummyConfig(device="cpu")
    runtime = config.resolve_runtime()
    assert runtime.device.type == "cpu"
    assert runtime.collector_backend == "multi_sync"
    assert runtime.single_env_backend == "n/a"
    assert runtime.collector_workers == 8


def test_runtime_config_request_and_runtime_dataclass_smoke():
    cfg = TorchRLRuntimeConfig(
        device="cpu",
        collector_backend="auto",
        single_env_backend="auto",
        collector_workers=None,
    )
    assert cfg.runtime_num_envs() == 1
    request = cfg.runtime_request()
    assert isinstance(request, TorchRLRuntimeRequest)
    assert request.num_envs == 1
    assert request.device == "cpu"

    runtime = TorchRLRuntime(
        device=torch.device("cpu"),
        collector_backend="single",
        single_env_backend="serial",
        collector_workers=None,
    )
    assert runtime.device.type == "cpu"
    assert runtime.collector_backend == "single"
    assert runtime.single_env_backend == "serial"
    assert runtime.collector_workers is None
