import pytest


def test_sac_eval_noise_mode_invalid_rejected_before_env_build(monkeypatch):
    from rl.torchrl.sac import trainer as torchrl_sac

    monkeypatch.setattr(
        torchrl_sac,
        "build_env_setup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("env build should not run")),
    )
    with pytest.raises(ValueError, match="eval_noise_mode must be one of"):
        torchrl_sac.train_sac(torchrl_sac.SACConfig(eval_noise_mode="invalid-mode"))
