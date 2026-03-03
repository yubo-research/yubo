import rl.torchrl.wpo as torchrl_wpo


def test_wpo_config_from_dict_converts_hidden_sizes():
    cfg = torchrl_wpo.WPOConfig.from_dict(
        {
            "exp_dir": "_tmp/wpo_test",
            "backbone_hidden_sizes": [128, 64],
        }
    )
    assert cfg.exp_dir == "_tmp/wpo_test"
    assert cfg.backbone_hidden_sizes == (128, 64)


def test_wpo_register_delegates_to_registry(monkeypatch):
    calls = []

    def fake_register_algo(name, config_cls, train_fn):
        calls.append((name, config_cls, train_fn))

    monkeypatch.setattr("rl.registry.register_algo", fake_register_algo)
    torchrl_wpo.register()

    assert len(calls) == 1
    name, config_cls, train_fn = calls[0]
    assert name == "wpo"
    assert config_cls is torchrl_wpo.WPOConfig
    assert train_fn is torchrl_wpo.train_wpo
