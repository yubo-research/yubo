from __future__ import annotations

import types

import pytest


def test_text_obj_runtime_base_seed_prefers_noise_seed():
    from problems.text_obj_runtime import base_seed

    cfg = types.SimpleNamespace(noise_seed_0=7, problem_seed=3, seed_offset=2)
    assert base_seed(cfg) == 9


def test_text_obj_runtime_base_seed_falls_back_to_problem_seed():
    from problems.text_obj_runtime import base_seed

    cfg = types.SimpleNamespace(noise_seed_0=None, problem_seed=3, seed_offset=1)
    assert base_seed(cfg) == 4


def test_text_obj_runtime_base_seed_treats_zero_noise_seed_as_explicit():
    from problems.text_obj_runtime import base_seed

    cfg = types.SimpleNamespace(noise_seed_0=0, problem_seed=9, seed_offset=1)
    assert base_seed(cfg) == 1


def test_text_obj_runtime_base_seed_defaults_to_zero():
    from problems.text_obj_runtime import base_seed

    cfg = types.SimpleNamespace(noise_seed_0=None, problem_seed=None, seed_offset=5)
    assert base_seed(cfg) == 5


def test_text_obj_runtime_require_runtime_lists_missing_modules(monkeypatch):
    from problems.text_obj_runtime import require_runtime

    monkeypatch.setattr(
        "problems.text_obj_runtime.importlib.util.find_spec",
        lambda name: None if name in {"ray", "vllm"} else object(),
    )

    with pytest.raises(RuntimeError, match="Missing: ray, vllm"):
        require_runtime()


def test_text_obj_runtime_make_adapter_root_uses_shm_when_writable(monkeypatch, tmp_path):
    from problems.text_obj_runtime import make_adapter_root

    shm = tmp_path / "shm"
    shm.mkdir()
    monkeypatch.setattr("problems.text_obj_runtime.os.path.isdir", lambda path: path == "/dev/shm")
    monkeypatch.setattr("problems.text_obj_runtime.os.access", lambda path, mode: path == "/dev/shm")
    monkeypatch.setattr("problems.text_obj_runtime.tempfile.mkdtemp", lambda prefix, dir: str(shm / "adapter"))

    assert make_adapter_root() == str(shm / "adapter")


def test_text_obj_runtime_make_adapter_root_falls_back_without_shm(monkeypatch, tmp_path):
    from problems.text_obj_runtime import make_adapter_root

    monkeypatch.setattr("problems.text_obj_runtime.os.path.isdir", lambda path: False)
    monkeypatch.setattr(
        "problems.text_obj_runtime.tempfile.mkdtemp",
        lambda prefix, dir: str(tmp_path / "adapter"),
    )

    assert make_adapter_root() == str(tmp_path / "adapter")
