import importlib


def test_sac_config_from_dict_converts_hidden_sizes():
    cfg_mod = importlib.import_module("rl.torchrl.sac.config")
    SACConfig = cfg_mod.SACConfig
    cfg = SACConfig.from_dict(
        {
            "env_tag": "cheetah",
            "exp_dir": "_tmp/sac_test",
            "backbone_hidden_sizes": [128, 64],
        }
    )
    assert cfg.exp_dir == "_tmp/sac_test"
    assert cfg.backbone_hidden_sizes == (128, 64)


def test_sac_config_from_dict_uses_env_defaults():
    cfg_mod = importlib.import_module("rl.torchrl.sac.config")
    SACConfig = cfg_mod.SACConfig
    cfg = SACConfig.from_dict({"env_tag": "cheetah"})
    assert cfg.backbone_hidden_sizes == (256, 256)


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
