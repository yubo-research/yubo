from types import SimpleNamespace

import pytest

from rl import runner
from rl.config_model_defaults import resolve_ppo_model_settings, resolve_sac_model_settings
from rl.core.grouped_config import dataclass_field_names, parse_dataclass_section
from rl.core.rl_video_settings import RLVideoSettings, attach_video_settings, get_video_settings, pop_video_settings
from rl.core.torchrl_collectors import collector_class
from rl.torchrl.ppo.config import PPOCheckpointConfig, PPOCollectorConfig, PPOConfig, PPOEvalConfig, PPOLossConfig, PPOOptimConfig, PPOProfileConfig
from rl.torchrl.sac.config import (
    SACCheckpointConfig,
    SACCollectorConfig,
    SACConfig,
    SACEvalConfig,
    SACLossConfig,
    SACOptimConfig,
    SACReplayBufferConfig,
    SACTargetNetUpdaterConfig,
)

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
_RUNTIME_FLAT_FIELDS = {
    "collector_backend",
    "single_env_backend",
    "collector_workers",
    "num_envs",
    "frames_per_batch",
}


@pytest.mark.parametrize("config_cls", [PPOConfig, SACConfig])
def test_rl_algorithm_configs_do_not_expose_model_or_video_fields(config_cls):
    fields = set(config_cls.__dataclass_fields__)
    assert fields.isdisjoint(_MODEL_FIELDS)
    assert fields.isdisjoint(_VIDEO_FIELDS)
    assert fields.isdisjoint(_RUNTIME_FLAT_FIELDS)
    assert "policy_tag" in fields
    assert "collector" in fields


def test_sac_config_uses_grouped_public_sections():
    cfg = SACConfig.from_dict(
        {
            "env_tag": "cheetah",
            "policy_tag": "mlp-32-16",
            "collector": {"num_envs": 8, "frames_per_batch": 64},
            "replay_buffer": {"batch_size": 128},
            "optim": {"actor_lr": 0.001},
            "loss": {"target_entropy": -6.0},
        }
    )
    dumped = cfg.to_dict()
    assert dumped["collector"]["num_envs"] == 8
    assert dumped["replay_buffer"]["batch_size"] == 128
    assert dumped["optim"]["actor_lr"] == 0.001
    assert dumped["loss"]["target_entropy"] == -6.0
    assert "learning_rate_actor" not in dumped
    assert cfg.collector.num_envs == 8
    assert cfg.optim.actor_lr == 0.001


def test_ppo_config_uses_grouped_public_sections():
    cfg = PPOConfig.from_dict(
        {
            "env_tag": "cheetah",
            "policy_tag": "mlp-32-16",
            "collector": {"num_envs": 2, "frames_per_batch": 128},
            "optim": {"lr": 0.001, "minibatch_size": 32},
            "loss": {"clip_epsilon": 0.1},
        }
    )
    dumped = cfg.to_dict()
    assert dumped["collector"]["frames_per_batch"] == 128
    assert dumped["optim"]["minibatch_size"] == 32
    assert dumped["loss"]["clip_epsilon"] == 0.1
    assert "learning_rate" not in dumped
    assert cfg.collector.frames_per_batch // cfg.collector.num_envs == 64
    assert cfg.collector.frames_per_batch // cfg.optim.minibatch_size == 4


def test_grouped_config_section_parser_validates_section_shape():
    assert "num_envs" in dataclass_field_names(SACCollectorConfig)
    assert parse_dataclass_section({"collector": {"num_envs": 3}}, "collector", SACCollectorConfig, label="SAC").num_envs == 3
    assert parse_dataclass_section({"collector": None}, "collector", SACCollectorConfig, label="SAC").num_envs == 1
    with pytest.raises(ValueError, match="must be a table"):
        parse_dataclass_section({"collector": True}, "collector", SACCollectorConfig, label="SAC")
    with pytest.raises(ValueError, match="Unknown SAC config fields"):
        parse_dataclass_section({"collector": {"bad": 1}}, "collector", SACCollectorConfig, label="SAC")


def test_grouped_config_section_defaults_are_explicit():
    assert SACCollectorConfig().total_frames == 1_000_000
    assert SACReplayBufferConfig().batch_size == 256
    assert SACOptimConfig().actor_lr == 0.0003
    assert SACLossConfig().gamma == 0.99
    assert SACTargetNetUpdaterConfig().tau == 0.005
    assert SACEvalConfig().interval_steps == 10000
    assert SACCheckpointConfig().resume_from is None
    assert PPOCollectorConfig().frames_per_batch == 2048
    assert PPOOptimConfig().minibatch_size == 64
    assert PPOLossConfig().clip_epsilon == 0.2
    assert PPOEvalConfig().interval == 1
    assert PPOCheckpointConfig().interval is None
    assert PPOProfileConfig().active == 3


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


@pytest.mark.parametrize("config_cls", [PPOConfig, SACConfig])
def test_isaaclab_configs_use_native_batched_single_collector(config_cls):
    cfg = config_cls.from_dict(
        {
            "env_tag": "isaaclab:Isaac-Fake-v0",
            "policy_tag": "mlp-32-16",
            "collector": {"num_envs": 2048, "backend": "auto"},
        }
    )

    request = cfg.runtime_request()

    assert request.collector_backend == "single"
    assert request.single_env_backend == "serial"
    assert request.num_envs == 2048


def test_collect_env_uses_torchrl_isaaclab_wrapper_for_isaaclab(monkeypatch):
    import rl.torchrl.collect_utils as cu

    calls = {}
    raw_env = object()

    def _make_raw(env_tag, **kwargs):
        calls["raw"] = (env_tag, kwargs)
        return raw_env

    class _Wrapper:
        def __init__(self, env, **kwargs):
            calls["wrapper"] = (env, kwargs)

    monkeypatch.setattr(cu, "make_raw_isaaclab_env", _make_raw)
    monkeypatch.setattr(cu.tr_envs, "IsaacLabWrapper", _Wrapper, raising=False)
    monkeypatch.setattr(
        cu.tr_envs,
        "TransformedEnv",
        lambda wrapped, transform: SimpleNamespace(wrapped=wrapped, transform=transform),
    )
    monkeypatch.setattr(cu.tr_transforms, "RenameTransform", lambda *a, **k: ("rename", a, k))
    monkeypatch.setattr(cu.tr_transforms, "DoubleToFloat", lambda: "double_to_float")
    monkeypatch.setattr(cu.tr_transforms, "Compose", lambda *items: ("compose", items))

    env_conf = SimpleNamespace(env_name="isaaclab:Isaac-Fake-v0", problem_seed=10, kwargs={"custom": 3})
    out = cu.make_collect_env(env_conf, env_index=2, num_envs=128, device="cuda:0")

    assert calls["raw"] == (
        "isaaclab:Isaac-Fake-v0",
        {"seed": 12, "num_envs": 128, "device": "cuda:0", "custom": 3},
    )
    assert calls["wrapper"][0] is raw_env
    assert str(calls["wrapper"][1]["device"]) == "cuda:0"
    assert out.transform[0] == "compose"
