import pytest

from rl.config_model_defaults import (
    apply_ppo_env_model_defaults,
    apply_sac_env_model_defaults,
)


def test_apply_ppo_env_model_defaults_uses_env_defaults():
    cfg = apply_ppo_env_model_defaults({"env_tag": "cheetah"})
    assert cfg["backbone_hidden_sizes"] == (64, 64)
    assert cfg["share_backbone"] is True


def test_apply_ppo_env_model_defaults_rejects_value_head_alias():
    with pytest.raises(ValueError, match="canonical key 'critic_head_hidden_sizes'"):
        _ = apply_ppo_env_model_defaults(
            {
                "env_tag": "cheetah",
                "value_head_hidden_sizes": [16],
            }
        )


def test_apply_env_model_defaults_requires_env_tag():
    with pytest.raises(ValueError, match="must set a non-empty 'env_tag'"):
        _ = apply_ppo_env_model_defaults({})
    with pytest.raises(ValueError, match="must set a non-empty 'env_tag'"):
        _ = apply_sac_env_model_defaults({})


def test_apply_sac_env_model_defaults_uses_env_defaults():
    cfg = apply_sac_env_model_defaults({"env_tag": "cheetah"})
    assert cfg["backbone_hidden_sizes"] == (256, 256)
    assert cfg["backbone_activation"] == "relu"
