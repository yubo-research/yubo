from __future__ import annotations

import problems.isaaclab_env_adapters as mod


def test_resolve_isaaclab_sim_device_honors_cpu(monkeypatch):
    monkeypatch.delenv("YUBO_ISAACLAB_DEVICE", raising=False)
    assert mod.resolve_isaaclab_sim_device("cpu") == "cpu"


def test_resolve_isaaclab_sim_device_falls_back_without_cuda(monkeypatch):
    monkeypatch.delenv("YUBO_ISAACLAB_DEVICE", raising=False)
    monkeypatch.setattr(mod, "_torch_cuda_usable", lambda: False)
    assert mod.resolve_isaaclab_sim_device("cuda") == "cpu"
    assert mod.resolve_isaaclab_sim_device("auto") == "cpu"


def test_resolve_isaaclab_sim_device_uses_cuda_when_available(monkeypatch):
    monkeypatch.delenv("YUBO_ISAACLAB_DEVICE", raising=False)
    monkeypatch.setattr(mod, "_torch_cuda_usable", lambda: True)
    monkeypatch.setattr(mod, "_nvidia_visible_devices_disabled", lambda: False)
    assert mod.resolve_isaaclab_sim_device("cuda") == "cuda:0"


def test_resolve_isaaclab_sim_device_env_override(monkeypatch):
    monkeypatch.setenv("YUBO_ISAACLAB_DEVICE", "cpu")
    monkeypatch.setattr(mod, "_torch_cuda_usable", lambda: True)
    assert mod.resolve_isaaclab_sim_device("cuda") == "cpu"
