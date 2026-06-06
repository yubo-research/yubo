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


def test_policy_presets_coverage():
    from policies.registry import POLICY_PRESETS

    assert "linear" in POLICY_PRESETS
    assert "pure-function" in POLICY_PRESETS
    assert "mlp-16-8" in POLICY_PRESETS
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


def test_get_policy_preset_dynamic_malformed_tags():
    from policies.registry import get_policy_preset

    with pytest.raises(KeyError, match="Unknown policy tag"):
        get_policy_preset("actor-mlp-")

    with pytest.raises(KeyError, match="Unknown policy tag"):
        get_policy_preset("mlp-0-16")


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
