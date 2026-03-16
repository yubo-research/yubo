import rl.sac as puffer_sac


def test_sac_config_from_dict_converts_hidden_sizes():
    cfg = puffer_sac.SACConfig.from_dict(
        {
            "env_tag": "cheetah",
            "exp_dir": "_tmp/sac_test",
            "backbone_hidden_sizes": [128, 64],
        }
    )
    assert cfg.exp_dir == "_tmp/sac_test"
    assert cfg.backbone_hidden_sizes == (128, 64)


def test_sac_config_from_dict_uses_env_defaults():
    cfg = puffer_sac.SACConfig.from_dict({"env_tag": "cheetah"})
    assert cfg.backbone_hidden_sizes == (256, 256)


def test_sac_register_delegates_to_registry():
    from rl import builtins
    from rl.registry import get_algo

    builtins.register_all()
    spec = get_algo("sac")
    assert spec.config_cls is puffer_sac.SACConfig
    assert spec.train_fn is puffer_sac.train_sac_puffer
