"""Tests for policies/registry.py module."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def test_policy_preset_dataclass_construction():
    from policies.registry import PolicyPreset

    def dummy_factory(env_runtime):
        return None

    preset = PolicyPreset(factory=dummy_factory)
    assert preset.factory is dummy_factory
    assert preset.rl_model is None

    rl_model = {"ppo": {"lr": 0.001}}
    preset_with_rl = PolicyPreset(factory=dummy_factory, rl_model=rl_model)
    assert preset_with_rl.rl_model == rl_model


def test_get_policy_preset_valid_tag():
    from policies.registry import get_policy_preset

    preset = get_policy_preset("linear")
    assert preset is not None
    assert callable(preset.factory)

    preset_mlp = get_policy_preset("mlp-16-8")
    assert preset_mlp.rl_model is not None
    assert "ppo" in preset_mlp.rl_model
    assert "sac" in preset_mlp.rl_model


def test_get_policy_preset_invalid_tag():
    from policies.registry import get_policy_preset

    with pytest.raises(KeyError, match="Unknown policy tag"):
        get_policy_preset("nonexistent-policy-xyz")


def test_get_policy_preset_pretrain_external_tags():
    from policies.registry import get_policy_preset, list_policy_tags

    for tag in ("hyperscalees-rwkv-7w3b-lora-r1", "nanoegg:int8:6l:256d"):
        preset = get_policy_preset(tag)
        assert preset is not None
        assert callable(preset.factory)
        assert tag in list_policy_tags()


def test_nanoegg_policy_tag_builds_nanoegg_policy_without_env_spaces():
    from problems.problem import build_problem

    problem = build_problem("pretrain:nanoegg:synthetic", "nanoegg:int8:1l:8d", problem_seed=0)
    policy = problem.build_policy()

    assert policy.is_nanoegg_pretrain_policy is True
    assert policy.env_name == "pretrain:nanoegg:synthetic"
    assert policy.policy_tag == "nanoegg:int8:1l:8d"
    assert policy.num_params() == 4096


def test_list_policy_tags_returns_sorted():
    from policies.registry import list_policy_tags

    tags = list_policy_tags()
    assert isinstance(tags, list)
    assert len(tags) > 0
    assert tags == sorted(tags)
    assert "linear" in tags
    assert "pure-function" in tags
    assert "mlp-16-8" in tags


def test_linear_factory():
    from policies.registry import _linear_factory

    env_runtime = SimpleNamespace(
        problem_seed=0,
        env_name="test",
        state_space=SimpleNamespace(shape=(4,)),
        action_space=SimpleNamespace(shape=(2,)),
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(4,))),
    )
    policy = _linear_factory(env_runtime)
    assert policy is not None
    assert hasattr(policy, "num_params")


def test_pure_function_factory():
    from policies.registry import _pure_function_factory

    env_runtime = SimpleNamespace(
        problem_seed=0,
        env_name="f:sphere-2d",
        state_space=SimpleNamespace(shape=(1,)),
        action_space=SimpleNamespace(low=np.array([-1, -1]), high=np.array([1, 1])),
        gym_conf=None,
    )
    policy = _pure_function_factory(env_runtime)
    assert policy is not None
    assert policy.num_params() == 2


def test_mlp_factory():
    from policies.registry import _mlp_factory

    factory = _mlp_factory((16, 8))
    assert callable(factory)

    env_runtime = SimpleNamespace(
        problem_seed=0,
        env_name="test",
        state_space=SimpleNamespace(shape=(4,)),
        action_space=SimpleNamespace(shape=(2,)),
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(4,))),
    )
    policy = factory(env_runtime)
    assert policy is not None
    assert hasattr(policy, "num_params")


def test_cnn_mlp_factory():
    from policies.registry import _cnn_mlp_factory

    factory = _cnn_mlp_factory((32, 16), cnn_latent_dim=64)
    assert callable(factory)

    env_runtime = SimpleNamespace(
        problem_seed=0,
        env_name="dm_control/cheetah-run-v0",
        state_space=SimpleNamespace(shape=(84, 84, 3)),
        action_space=SimpleNamespace(shape=(6,)),
        gym_conf=None,
    )
    policy = factory(env_runtime)
    assert policy is not None
    assert hasattr(policy, "num_params")
    assert policy.num_params() > 0


def test_infer_rl_model_from_mlp():
    from policies.registry import _infer_rl_model_from_mlp

    result = _infer_rl_model_from_mlp((64, 32))
    assert "ppo" in result
    assert "sac" in result
    assert result["ppo"]["backbone_hidden_sizes"] == (64, 32)
    assert result["sac"]["backbone_hidden_sizes"] == (64, 32)
    assert result["ppo"]["backbone_name"] == "mlp"
    assert result["ppo"]["backbone_activation"] == "silu"
    assert result["ppo"]["backbone_layer_norm"] is True
    assert result["ppo"]["share_backbone"] is True
    assert result["ppo"]["log_std_init"] == -0.5


def test_infer_rl_model_from_actor_critic_mlp():
    from policies.registry import _infer_rl_model_from_actor_critic_mlp

    result = _infer_rl_model_from_actor_critic_mlp((16, 8))
    assert "ppo" in result
    assert "sac" in result
    assert result["ppo"]["backbone_hidden_sizes"] == (16, 8)
    assert result["ppo"]["log_std_init"] == 0.0


def test_dynamic_actor_critic_mlp_policy_tag_allows_activation():
    from policies.registry import get_policy_preset

    preset = get_policy_preset("actor-critic-mlp-256-256-tanh")
    assert preset.rl_model["ppo"]["backbone_hidden_sizes"] == (256, 256)
    assert preset.rl_model["ppo"]["backbone_activation"] == "tanh"
    assert preset.rl_model["ppo"]["head_activation"] == "tanh"
    assert preset.rl_model["ppo"]["backbone_layer_norm"] is True


def test_sb3_ppo_mlp_policy_tag_builds_matching_actor():
    from policies.actor_critic_mlp_policy import ActorCriticMLPPolicy
    from policies.registry import get_policy_preset

    preset = get_policy_preset("sb3-ppo-mlp-64-64-tanh")
    env_runtime = SimpleNamespace(
        problem_seed=0,
        env_name="HalfCheetah-v5",
        state_space=SimpleNamespace(shape=(17,)),
        action_space=SimpleNamespace(shape=(6,)),
        gym_conf=None,
    )
    policy = preset.factory(env_runtime)

    assert isinstance(policy, ActorCriticMLPPolicy)
    assert policy.actor_backbone[0].weight.shape == (64, 17)
    assert policy.actor_backbone[2].weight.shape == (64, 64)
    assert policy.actor_head.weight.shape == (6, 64)
    assert preset.rl_model is None


def test_policy_presets_coverage():
    from policies.registry import POLICY_PRESETS

    assert "linear" in POLICY_PRESETS
    assert "pure-function" in POLICY_PRESETS
    assert "mlp-16-8" in POLICY_PRESETS
    assert "cnn-mlp-1024-600" in POLICY_PRESETS
    assert "bipedal-heuristic" in POLICY_PRESETS
    assert "turbo-lunar" in POLICY_PRESETS
    assert "actor-critic-mlp-16-8" in POLICY_PRESETS

    for tag, preset in POLICY_PRESETS.items():
        assert callable(preset.factory), f"Factory for {tag} not callable"


def test_get_policy_preset_dynamic_actor_mlp():
    from policies.actor_mlp_policy import ActorMLPPolicy
    from policies.registry import get_policy_preset

    preset_32_16 = get_policy_preset("actor-mlp-32-16")
    preset_32_32 = get_policy_preset("actor-mlp-32-32")

    env_runtime = SimpleNamespace(
        problem_seed=0,
        env_name="test",
        state_space=SimpleNamespace(shape=(4,)),
        action_space=SimpleNamespace(shape=(2,)),
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(4,))),
    )
    policy_32_16 = preset_32_16.factory(env_runtime)
    policy_32_32 = preset_32_32.factory(env_runtime)

    assert isinstance(policy_32_16, ActorMLPPolicy)
    assert isinstance(policy_32_32, ActorMLPPolicy)
    assert policy_32_16.num_params() != policy_32_32.num_params()


def test_get_policy_preset_dynamic_mlp_unregistered_size():
    from policies.mlp_policy import MLPPolicy
    from policies.registry import get_policy_preset

    preset = get_policy_preset("mlp-7-3")
    env_runtime = SimpleNamespace(
        problem_seed=0,
        env_name="test",
        state_space=SimpleNamespace(shape=(4,)),
        action_space=SimpleNamespace(shape=(2,)),
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(4,))),
    )
    policy = preset.factory(env_runtime)
    assert isinstance(policy, MLPPolicy)


@pytest.mark.parametrize(
    "prefix,bad_tag,good_tag,expected_sizes",
    [
        ("mlp-", "mlp-032-016", "mlp-32-16", (32, 16)),
        ("mlp-", "mlp-007-16", "mlp-7-16", (7, 16)),
        ("mlp-", "mlp-16-08", "mlp-16-8", (16, 8)),
        ("actor-mlp-", "actor-mlp-08-16", "actor-mlp-8-16", (8, 16)),
        ("actor-critic-mlp-", "actor-critic-mlp-032-032", "actor-critic-mlp-32-32", (32, 32)),
    ],
)
def test_parse_sizes_suffix_rejects_leading_zero_segments(prefix, bad_tag, good_tag, expected_sizes):
    from policies.registry import _parse_sizes_suffix

    assert _parse_sizes_suffix(prefix, bad_tag) is None
    assert _parse_sizes_suffix(prefix, good_tag) == expected_sizes


def test_get_policy_preset_dynamic_malformed_tags():
    from policies.registry import get_policy_preset

    with pytest.raises(KeyError, match="Unknown policy tag"):
        get_policy_preset("actor-mlp-")

    with pytest.raises(KeyError, match="Unknown policy tag"):
        get_policy_preset("mlp-0-16")

    with pytest.raises(KeyError, match="Unknown policy tag"):
        get_policy_preset("mlp-032-016")

    with pytest.raises(KeyError, match="Unknown policy tag"):
        get_policy_preset("mlp-007-16")

    with pytest.raises(KeyError, match="Unknown policy tag"):
        get_policy_preset("actor-mlp-08-16")


def test_get_policy_preset_static_mlp_regression():
    from policies.registry import get_policy_preset

    preset = get_policy_preset("mlp-32-16")
    assert preset.rl_model is not None
    assert preset.rl_model["ppo"]["backbone_hidden_sizes"] == (32, 16)


def test_build_problem_dynamic_actor_mlp_cheetah():
    from problems.problem import build_problem

    problem = build_problem("cheetah", "actor-mlp-32-16", problem_seed=0, noise_seed_0=0)
    policy = problem.build_policy()
    assert hasattr(policy, "get_action_and_value")
    assert hasattr(policy, "last_log_probs")
