import pytest


def test_sac_eval_noise_mode_invalid_rejected_before_env_build():
    from rl.eval_noise import normalize_eval_noise_mode

    with pytest.raises(ValueError, match="eval_noise_mode must be one of"):
        normalize_eval_noise_mode("invalid-mode")
