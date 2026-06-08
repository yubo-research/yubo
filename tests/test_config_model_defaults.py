import pytest

from rl.config_model_defaults import (
    apply_ppo_env_model_defaults,
    apply_sac_env_model_defaults,
    reject_model_config_keys,
    resolve_ppo_model_settings,
    resolve_sac_model_settings,
)


def test_apply_ppo_env_model_defaults_uses_env_defaults():
    cfg = apply_ppo_env_model_defaults({"env_tag": "cheetah", "policy_tag": "mlp-32-16"})
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
    cfg = apply_sac_env_model_defaults({"env_tag": "cheetah", "policy_tag": "mlp-32-16"})
    assert cfg["backbone_hidden_sizes"] == (256, 256)
    assert cfg["backbone_activation"] == "relu"


def test_reject_model_config_keys_allows_public_runtime_keys():
    reject_model_config_keys({"env_tag": "cheetah", "policy_tag": "mlp-32-16", "num_envs": 8}, algo="sac")


def test_reject_model_config_keys_reports_all_explicit_model_keys():
    with pytest.raises(ValueError, match="backbone_name, theta_dim"):
        reject_model_config_keys({"env_tag": "cheetah", "backbone_name": "mlp", "theta_dim": 10}, algo="sac")


def test_resolve_model_settings_requires_policy_tag():
    config = type("_Cfg", (), {"env_tag": "pend", "policy_tag": None})()
    with pytest.raises(ValueError, match="Missing required argument 'policy_tag'"):
        resolve_ppo_model_settings(config)
    with pytest.raises(ValueError, match="Missing required argument 'policy_tag'"):
        resolve_sac_model_settings(config)
