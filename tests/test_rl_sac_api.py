from rl.algos.backends.torchrl.sac import api as torchrl_sac


def test_sac_config_from_dict_converts_hidden_sizes():
    cfg = torchrl_sac.SACConfig.from_dict(
        {
            "exp_dir": "_tmp/sac_test",
            "backbone_hidden_sizes": [128, 64],
        }
    )
    assert cfg.exp_dir == "_tmp/sac_test"
    assert cfg.backbone_hidden_sizes == (128, 64)


def test_sac_register_delegates_to_registry(monkeypatch):
    calls = []

    def fake_register_algo(name, config_cls, train_fn):
        calls.append((name, config_cls, train_fn))

    monkeypatch.setattr("rl.algos.registry.register_algo", fake_register_algo)
    torchrl_sac.register()

    assert len(calls) == 1
    name, config_cls, train_fn = calls[0]
    assert name == "sac"
    assert config_cls is torchrl_sac.SACConfig
    assert train_fn is torchrl_sac.train_sac
