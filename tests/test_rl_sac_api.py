import importlib
from types import SimpleNamespace

import pytest


def test_sac_config_from_dict_rejects_explicit_model_fields():
    cfg_mod = importlib.import_module("rl.torchrl.sac.config")
    SACConfig = cfg_mod.SACConfig
    with pytest.raises(ValueError, match="policy_tag for model architecture"):
        SACConfig.from_dict({"env_tag": "cheetah", "policy_tag": "mlp-32-16", "backbone_hidden_sizes": [128, 64]})


def test_sac_config_from_dict_keeps_public_runtime_fields():
    cfg_mod = importlib.import_module("rl.torchrl.sac.config")
    SACConfig = cfg_mod.SACConfig
    cfg = SACConfig.from_dict({"env_tag": "cheetah", "policy_tag": "mlp-32-16", "exp_dir": "_tmp/sac_test", "replay_pin_memory": True, "replay_prefetch": 2})
    assert cfg.exp_dir == "_tmp/sac_test"
    assert cfg.replay_pin_memory is True
    assert cfg.replay_prefetch == 2


def test_sac_config_from_dict_uses_env_defaults():
    from rl.config_model_defaults import resolve_sac_model_settings

    cfg_mod = importlib.import_module("rl.torchrl.sac.config")
    SACConfig = cfg_mod.SACConfig
    cfg = SACConfig.from_dict({"env_tag": "cheetah", "policy_tag": "mlp-32-16"})
    assert resolve_sac_model_settings(cfg).backbone_hidden_sizes == (256, 256)


def test_sac_register_delegates_to_registry(monkeypatch):
    sac = importlib.import_module("rl.torchrl.sac")
    cfg_mod = importlib.import_module("rl.torchrl.sac.config")
    SACConfig = cfg_mod.SACConfig
    train_sac = sac.train_sac
    calls = []

    def fake_register_algo(name, config_cls, train_fn):
        calls.append((name, config_cls, train_fn))

    monkeypatch.setattr("rl.registry.register_algo", fake_register_algo)
    sac.register()

    assert len(calls) == 1
    name, config_cls, train_fn = calls[0]
    assert name == "sac"
    assert config_cls is SACConfig
    assert train_fn is train_sac


def test_sac_multi_collector_policy_sync_helper():
    phase_b = importlib.import_module("rl.torchrl.sac.sac_trainer_phase_b_impl")
    calls = {"n": 0}

    class _Collector:
        def update_policy_weights_(self):
            calls["n"] += 1

    phase_b.sync_collector_policy_if_needed(_Collector(), SimpleNamespace(collector_backend="multi_sync"))
    assert calls["n"] == 1

    phase_b.sync_collector_policy_if_needed(_Collector(), SimpleNamespace(collector_backend="single"))
    assert calls["n"] == 1
