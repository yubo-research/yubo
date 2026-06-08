import importlib


def test_sac_eval_noise_mode_invalid_rejected_before_env_build(monkeypatch):
    pytest = importlib.import_module("pytest")
    cfg_mod = importlib.import_module("rl.torchrl.sac.config")
    sac = importlib.import_module("rl.torchrl.sac")

    monkeypatch.setattr(
        "rl.torchrl.sac.sac_setup_build.build_env_setup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("env build should not run")),
    )
    with pytest.raises(ValueError, match="eval_noise_mode must be one of"):
        sac.train_sac(cfg_mod.SACConfig(eval=cfg_mod.SACEvalConfig(noise_mode="invalid-mode")))
