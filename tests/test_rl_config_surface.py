from types import SimpleNamespace

import pytest

from rl import runner
from rl.config_model_defaults import resolve_ppo_model_settings, resolve_sac_model_settings
from rl.core.rl_video_settings import RLVideoSettings, attach_video_settings, get_video_settings, pop_video_settings
from rl.core.torchrl_collectors import collector_class
from rl.torchrl.ppo.config import PPOConfig
from rl.torchrl.sac.config import SACConfig

_MODEL_FIELDS = {
    "backbone_name",
    "backbone_hidden_sizes",
    "backbone_activation",
    "backbone_layer_norm",
    "actor_head_hidden_sizes",
    "critic_head_hidden_sizes",
    "head_activation",
    "share_backbone",
    "log_std_init",
    "theta_dim",
}
_VIDEO_FIELDS = {
    "video_enable",
    "video_prefix",
    "video_num_episodes",
    "video_num_video_episodes",
    "video_episode_selection",
    "video_seed_base",
}


@pytest.mark.parametrize("config_cls", [PPOConfig, SACConfig])
def test_rl_algorithm_configs_do_not_expose_model_or_video_fields(config_cls):
    fields = set(config_cls.__dataclass_fields__)
    assert fields.isdisjoint(_MODEL_FIELDS)
    assert fields.isdisjoint(_VIDEO_FIELDS)
    assert "policy_tag" in fields


@pytest.mark.parametrize(
    ("config_cls", "bad_key"),
    [
        (PPOConfig, "share_backbone"),
        (PPOConfig, "backbone_hidden_sizes"),
        (SACConfig, "theta_dim"),
        (SACConfig, "critic_head_hidden_sizes"),
    ],
)
def test_rl_algorithm_configs_reject_explicit_model_fields(config_cls, bad_key):
    with pytest.raises(ValueError, match="policy_tag for model architecture"):
        config_cls.from_dict({"env_tag": "cheetah", "policy_tag": "mlp-32-16", bad_key: [32]})


def test_policy_tag_still_resolves_internal_model_settings():
    ppo = PPOConfig.from_dict({"env_tag": "cheetah", "policy_tag": "mlp-32-16"})
    sac = SACConfig.from_dict({"env_tag": "cheetah", "policy_tag": "mlp-32-16"})

    assert resolve_ppo_model_settings(ppo).backbone_hidden_sizes == (64, 64)
    assert resolve_sac_model_settings(sac).backbone_hidden_sizes == (256, 256)


def test_run_artifact_video_table_becomes_internal_video_settings(monkeypatch):
    captured = {}

    class _AlgoConfig(SACConfig):
        @classmethod
        def from_dict(cls, raw):
            config = super().from_dict(raw)
            captured["config"] = config
            return config

    monkeypatch.setattr(
        "rl.registry.get_algo",
        lambda _name: SimpleNamespace(config_cls=_AlgoConfig, train_fn=lambda config: config),
    )

    runner._run_from_cfg(
        {
            "rl": {
                "algo": "sac",
                "sac": {"env_tag": "cheetah", "policy_tag": "mlp-32-16"},
                "run": {
                    "artifacts": {
                        "video": {
                            "enable": True,
                            "prefix": "policy-check",
                            "num_video_episodes": 2,
                        }
                    }
                },
            }
        }
    )

    config = captured["config"]
    settings = get_video_settings(config)
    assert "video_enable" not in config.__dataclass_fields__
    assert settings.enable is True
    assert settings.prefix == "policy-check"
    assert settings.num_video_episodes == 2


def test_video_settings_reads_legacy_namespace_for_call_sites_not_yet_on_configs():
    settings = get_video_settings(
        SimpleNamespace(
            video_enable=True,
            video_prefix="legacy",
            video_num_episodes=4,
            video_num_video_episodes=1,
            video_episode_selection="first",
            video_seed_base=17,
        )
    )
    assert settings.enable is True
    assert settings.prefix == "legacy"
    assert settings.num_episodes == 4
    assert settings.num_video_episodes == 1
    assert settings.episode_selection == "first"
    assert settings.seed_base == 17


def test_video_settings_pop_and_attach_keep_video_out_of_public_config():
    raw = {
        "env_tag": "cheetah",
        "video_enable": True,
        "video_prefix": "rollout",
        "video_num_episodes": 5,
        "video_num_video_episodes": 2,
        "video_episode_selection": "best",
        "video_seed_base": 99,
    }
    settings = pop_video_settings(raw)
    config = attach_video_settings(SimpleNamespace(env_tag=raw["env_tag"]), settings)

    assert raw == {"env_tag": "cheetah"}
    assert get_video_settings(config) == RLVideoSettings(
        enable=True,
        prefix="rollout",
        num_episodes=5,
        num_video_episodes=2,
        episode_selection="best",
        seed_base=99,
    )


def test_collector_lookup_handles_torchrl_export_layout():
    assert collector_class("SyncDataCollector").__name__ == "SyncDataCollector"
    assert collector_class("MultiSyncDataCollector").__name__ == "MultiSyncDataCollector"
