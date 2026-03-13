import rl.torchrl.sac as torchrl_sac
from rl.torchrl.sac import setup as torchrl_sac_setup


def test_sac_config_from_dict_converts_hidden_sizes():
    cfg = torchrl_sac.SACConfig.from_dict(
        {
            "env_tag": "cheetah",
            "exp_dir": "_tmp/sac_test",
            "backbone_hidden_sizes": [128, 64],
        }
    )
    assert cfg.exp_dir == "_tmp/sac_test"
    assert cfg.backbone_hidden_sizes == (128, 64)


def test_sac_config_from_dict_uses_env_defaults():
    cfg = torchrl_sac.SACConfig.from_dict({"env_tag": "cheetah"})
    assert cfg.backbone_hidden_sizes == (256, 256)


def test_sac_register_delegates_to_registry(monkeypatch):
    calls = []

    def fake_register_algo(name, config_cls, train_fn):
        calls.append((name, config_cls, train_fn))

    monkeypatch.setattr("rl.registry.register_algo", fake_register_algo)
    torchrl_sac.register()

    assert len(calls) == 1
    name, config_cls, train_fn = calls[0]
    assert name == "sac"
    assert config_cls is torchrl_sac.SACConfig
    assert train_fn is torchrl_sac.train_sac


def test_sac_build_pipeline(monkeypatch, tmp_path):
    fake_env_conf = type(
        "EnvConf",
        (),
        {
            "env_name": "pend",
            "obs_mode": "vector",
            "state_space": type("S", (), {"shape": (3,)})(),
        },
    )()
    monkeypatch.setattr(
        torchrl_sac_setup,
        "build_continuous_env_setup",
        lambda **_kwargs: type(
            "Shared",
            (),
            {
                "env_conf": fake_env_conf,
                "problem_seed": 5,
                "noise_seed_0": 6,
                "act_dim": 2,
                "action_low": torchrl_sac_setup.np.array([-1.0, -1.0], dtype=torchrl_sac_setup.np.float32),
                "action_high": torchrl_sac_setup.np.array([1.0, 1.0], dtype=torchrl_sac_setup.np.float32),
                "obs_lb": None,
                "obs_width": None,
            },
        )(),
    )

    cfg = torchrl_sac.SACConfig(exp_dir=str(tmp_path), env_tag="pend", batch_size=4, replay_size=8)
    env = torchrl_sac_setup.build_env_setup(cfg)
    modules = torchrl_sac_setup.build_modules(cfg, env, device=torchrl_sac_setup.torch.device("cpu"))
    training = torchrl_sac_setup.build_training(cfg, modules)
    assert env.problem_seed == 5
    assert training.replay.batch_size == 4
    assert training.exp_dir == tmp_path
