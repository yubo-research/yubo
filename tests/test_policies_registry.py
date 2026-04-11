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
    assert "moe-16-16-4" in tags
    assert "gauss-rl-hardgate" in tags


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


def test_moe_factory():
    from policies.registry import _moe_factory

    factory = _moe_factory((16, 16), num_experts=4, router_hidden_sizes=(16,), top_k=2)
    assert callable(factory)

    env_runtime = SimpleNamespace(
        problem_seed=0,
        env_name="HalfCheetah-v5",
        state_space=SimpleNamespace(shape=(17,)),
        action_space=SimpleNamespace(shape=(6,)),
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(17,))),
    )
    policy = factory(env_runtime)
    assert policy is not None
    assert hasattr(policy, "num_params")
    assert policy._num_experts == 4
    assert policy._top_k == 2


def test_moe_registry_preset():
    from policies.registry import get_policy_preset

    env_runtime = SimpleNamespace(
        problem_seed=0,
        env_name="HalfCheetah-v5",
        state_space=SimpleNamespace(shape=(17,)),
        action_space=SimpleNamespace(shape=(6,)),
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(17,))),
    )
    policy = get_policy_preset("moe-16-16-4").factory(env_runtime)
    assert policy is not None
    assert policy._num_experts == 4
    assert policy._top_k == 2


def test_gauss_rl_gauss_tanh_preset_uses_tanh_clip():
    from policies.registry import get_policy_preset
    from rl.policy_backbone import GaussianActorBackbonePolicy

    env_runtime = SimpleNamespace(
        problem_seed=0,
        env_name="HalfCheetah-v5",
        state_space=SimpleNamespace(shape=(17,)),
        action_space=SimpleNamespace(shape=(6,)),
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(17,))),
        ensure_spaces=lambda: None,
    )

    policy = get_policy_preset("gauss-rl-gauss-tanh").factory(env_runtime)
    assert isinstance(policy, GaussianActorBackbonePolicy)
    assert policy._squash_mode == "tanh_clip"
    assert policy._deterministic_eval is True


def test_gauss_rl_gauss_tanh_stoch_preset_uses_stochastic_tanh_clip():
    from policies.registry import get_policy_preset
    from rl.policy_backbone import GaussianActorBackbonePolicy

    env_runtime = SimpleNamespace(
        problem_seed=0,
        env_name="HalfCheetah-v5",
        state_space=SimpleNamespace(shape=(17,)),
        action_space=SimpleNamespace(shape=(6,)),
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(17,))),
        ensure_spaces=lambda: None,
    )

    policy = get_policy_preset("gauss-rl-gauss-tanh-stoch").factory(env_runtime)
    assert isinstance(policy, GaussianActorBackbonePolicy)
    assert policy._squash_mode == "tanh_clip"
    assert policy._deterministic_eval is False


def test_gauss_rl_hardgate_preset_uses_hardgate_backbone():
    from policies.registry import get_policy_preset
    from rl.policy_backbone import GaussianActorBackbonePolicy
    from rl.shared_gaussian_actor import get_gaussian_actor_spec

    env_runtime = SimpleNamespace(
        problem_seed=0,
        env_name="HalfCheetah-v5",
        state_space=SimpleNamespace(shape=(17,)),
        action_space=SimpleNamespace(shape=(6,)),
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(17,))),
        ensure_spaces=lambda: None,
    )

    backbone, head = get_gaussian_actor_spec("rl-hardgate")
    assert backbone.name == "hardgate_residual_mlp"
    assert backbone.hidden_sizes == (16, 16)
    assert head.hidden_sizes == ()

    policy = get_policy_preset("gauss-rl-hardgate").factory(env_runtime)
    assert isinstance(policy, GaussianActorBackbonePolicy)
    assert policy._squash_mode == "tanh_clip"
    assert policy._deterministic_eval is True


def test_gauss_rl_hardgate_variant_presets():
    from policies.registry import get_policy_preset
    from rl.policy_backbone import GaussianActorBackbonePolicy
    from rl.shared_gaussian_actor import get_gaussian_actor_spec

    env_runtime = SimpleNamespace(
        problem_seed=0,
        env_name="HalfCheetah-v5",
        state_space=SimpleNamespace(shape=(17,)),
        action_space=SimpleNamespace(shape=(6,)),
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(17,))),
        ensure_spaces=lambda: None,
    )

    small_backbone, small_head = get_gaussian_actor_spec("rl-hardgate-small")
    assert small_backbone.hidden_sizes == (16, 8)
    assert small_backbone.activation == "silu"
    assert small_head.hidden_sizes == ()

    large_backbone, large_head = get_gaussian_actor_spec("rl-hardgate-large-tanh")
    assert large_backbone.hidden_sizes == (64, 64)
    assert large_backbone.activation == "tanh"
    assert large_head.hidden_sizes == ()

    small_policy = get_policy_preset("gauss-rl-hardgate-small").factory(env_runtime)
    assert isinstance(small_policy, GaussianActorBackbonePolicy)
    assert small_policy._deterministic_eval is True

    large_policy = get_policy_preset("gauss-rl-hardgate-large-tanh").factory(env_runtime)
    assert isinstance(large_policy, GaussianActorBackbonePolicy)
    assert large_policy._deterministic_eval is True

    stoch_policy = get_policy_preset("gauss-rl-hardgate-stoch").factory(env_runtime)
    assert isinstance(stoch_policy, GaussianActorBackbonePolicy)
    assert stoch_policy._deterministic_eval is False


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
    assert "moe-16-16-4" in POLICY_PRESETS
    assert "bipedal-heuristic" in POLICY_PRESETS
    assert "turbo-lunar" in POLICY_PRESETS
    assert "actor-critic-mlp-16-8" in POLICY_PRESETS
    assert "gauss-rl-hardgate" in POLICY_PRESETS
    assert "gauss-rl-hardgate-small" in POLICY_PRESETS
    assert "gauss-rl-hardgate-large-tanh" in POLICY_PRESETS
    assert "gauss-rl-hardgate-stoch" in POLICY_PRESETS

    for tag, preset in POLICY_PRESETS.items():
        assert callable(preset.factory), f"Factory for {tag} not callable"
