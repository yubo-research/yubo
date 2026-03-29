from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from rl.pufferlib.offpolicy import engine_utils, runtime_utils


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_engine_utils_init_run_artifacts(monkeypatch, tmp_path: Path):
    write_rec = _Recorder()

    class _CheckpointManager:
        def __init__(self, *, exp_dir):
            self.exp_dir = Path(exp_dir)

    def _fake_import_module(name: str):
        if name == "analysis.data_io":
            return SimpleNamespace(write_config=write_rec)
        if name == "rl.checkpointing":
            return SimpleNamespace(CheckpointManager=_CheckpointManager)
        raise AssertionError(name)

    monkeypatch.setattr(engine_utils.importlib, "import_module", _fake_import_module)

    exp_path, metrics_path, checkpoint_mgr = engine_utils.init_run_artifacts(
        exp_dir=str(tmp_path / "exp"),
        config_dict={"k": 1},
    )
    assert exp_path.exists()
    assert metrics_path == exp_path / "metrics.jsonl"
    assert checkpoint_mgr.exp_dir == exp_path
    assert len(write_rec.calls) == 1
    assert write_rec.calls[0][0][0] == str(exp_path)
    assert write_rec.calls[0][0][1] == {"k": 1}


def test_engine_utils_init_runtime(monkeypatch):
    def _fake_import_module(name: str):
        assert name == "rl.core.env_conf"
        return SimpleNamespace(global_seed_for_run=lambda seed: seed + 17)

    monkeypatch.setattr(engine_utils.importlib, "import_module", _fake_import_module)
    seed_calls = []

    cfg = SimpleNamespace(device="cpu")
    env_setup, device = engine_utils.init_runtime(
        cfg,
        build_env_setup_fn=lambda _cfg: SimpleNamespace(problem_seed=11, payload="ok"),
        seed_everything_fn=lambda seed: seed_calls.append(seed),
        resolve_device_fn=lambda device_name: f"resolved:{device_name}",
    )

    assert env_setup.payload == "ok"
    assert device == "resolved:cpu"
    assert seed_calls == [28]


def test_engine_utils_checkpoint_mark_if_due():
    saves = []
    mark = engine_utils.checkpoint_mark_if_due(
        global_step=5,
        checkpoint_interval_steps=10,
        previous_mark=0,
        due_mark_fn=lambda *_args, **_kwargs: None,
        save_fn=lambda: saves.append("saved"),
    )
    assert mark == 0
    assert saves == []

    mark = engine_utils.checkpoint_mark_if_due(
        global_step=20,
        checkpoint_interval_steps=10,
        previous_mark=1,
        due_mark_fn=lambda *_args, **_kwargs: 2,
        save_fn=lambda: saves.append("saved"),
    )
    assert mark == 2
    assert saves == ["saved"]


def test_runtime_utils_select_device_and_obs_scale(monkeypatch):
    calls = {}

    def _fake_select_device(device, *, cuda_is_available_fn, mps_is_available_fn):
        calls["device"] = device
        calls["cuda"] = bool(cuda_is_available_fn())
        calls["mps"] = bool(mps_is_available_fn())
        return torch.device("cpu")

    monkeypatch.setattr(runtime_utils, "_select_device_core", _fake_select_device)
    monkeypatch.setattr(runtime_utils, "_mps_is_available_core", lambda: True)
    monkeypatch.setattr(
        runtime_utils,
        "_obs_scale_from_env_core",
        lambda env_conf: ("lb", "w", env_conf),
    )

    out = runtime_utils.select_device("auto")
    assert out.type == "cpu"
    assert calls["device"] == "auto"
    assert isinstance(calls["cuda"], bool)
    assert calls["mps"] is True

    assert runtime_utils.obs_scale_from_env("env") == ("lb", "w", "env")
