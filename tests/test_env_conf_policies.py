from __future__ import annotations

import sys
from types import ModuleType

import pytest

from problems import env_conf_policies


def test_resolve_dm_control_policy_class_variants(monkeypatch):
    monkeypatch.setattr(env_conf_policies, "gaussian_policy_factory", lambda variant, **_kw: f"gauss:{variant}")

    with pytest.raises(ValueError, match="cnn_mlp_policy_factory is required"):
        env_conf_policies.resolve_dm_control_policy_class(use_pixels=True, policy_variant=None, cnn_mlp_policy_factory=None)

    px = env_conf_policies.resolve_dm_control_policy_class(
        use_pixels=True,
        policy_variant=None,
        cnn_mlp_policy_factory=lambda sizes: ("cnn", tuple(sizes)),
    )
    assert px == ("cnn", (32, 16))

    assert env_conf_policies.resolve_dm_control_policy_class(use_pixels=False, policy_variant="gauss") == "gauss:rl-gauss-tanh"
    assert env_conf_policies.resolve_dm_control_policy_class(use_pixels=False, policy_variant="rl-gauss") == "gauss:rl-gauss"

    default = env_conf_policies.resolve_dm_control_policy_class(use_pixels=False, policy_variant=None)
    assert default.__class__.__name__ == "MLPPolicyFactory"


def test_resolve_atari_policy_class_variants(monkeypatch):
    fake_policy_backbone = ModuleType("rl.policy_backbone")

    class _FakeMLP16:
        pass

    fake_policy_backbone.AtariMLP16DiscretePolicy = _FakeMLP16
    monkeypatch.setitem(sys.modules, "rl.policy_backbone", fake_policy_backbone)

    assert env_conf_policies.resolve_atari_policy_class(policy_variant="mlp16") is _FakeMLP16

    with pytest.raises(ValueError, match="atari_agent57_factory is required"):
        env_conf_policies.resolve_atari_policy_class(policy_variant="agent57")

    with pytest.raises(ValueError, match="atari_gaussian_policy_factory is required"):
        env_conf_policies.resolve_atari_policy_class(policy_variant="gauss")

    with pytest.raises(ValueError, match="atari_cnn_policy_factory is required"):
        env_conf_policies.resolve_atari_policy_class(policy_variant=None)

    agent57 = env_conf_policies.resolve_atari_policy_class(
        policy_variant="agent57",
        atari_agent57_factory=lambda **kwargs: ("agent57", kwargs),
    )
    assert agent57[0] == "agent57"
    assert agent57[1]["cnn_variant"] == "small"

    gauss = env_conf_policies.resolve_atari_policy_class(
        policy_variant="gauss",
        atari_gaussian_policy_factory=lambda **kwargs: ("gauss", kwargs),
    )
    assert gauss[0] == "gauss"
    assert gauss[1]["variant"] == "small"

    default = env_conf_policies.resolve_atari_policy_class(
        policy_variant=None,
        atari_cnn_policy_factory=lambda sizes, **kwargs: ("cnn", tuple(sizes), kwargs),
    )
    assert default[0] == "cnn"
    assert default[1] == (24,)


def test_gaussian_policy_factory_import(monkeypatch):
    fake_policy_backbone = ModuleType("rl.policy_backbone")

    class _FakeFactory:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_policy_backbone.GaussianActorBackbonePolicyFactory = _FakeFactory
    monkeypatch.setitem(sys.modules, "rl.policy_backbone", fake_policy_backbone)

    factory = env_conf_policies.gaussian_policy_factory("rl-gauss", foo=1)
    assert isinstance(factory, _FakeFactory)
    assert factory.kwargs["variant"] == "rl-gauss"
    assert factory.kwargs["foo"] == 1


def test_load_atari_dm_bindings_with_fake_modules(monkeypatch):
    from problems import env_conf_backends

    fake_atari_env = ModuleType("problems.atari_env")

    class _FakeAtariPreprocessOptions:
        pass

    fake_atari_env.AtariPreprocessOptions = _FakeAtariPreprocessOptions
    fake_atari_env._parse_atari_tag = lambda tag: "ALE/Pong-v5" if "Pong" in str(tag) else str(tag)
    fake_atari_env.make = lambda *args, **kwargs: ("atari-make", args, kwargs)

    fake_dm_env = ModuleType("problems.dm_control_env")
    fake_dm_env.make = lambda *args, **kwargs: ("dm-make", args, kwargs)

    fake_pixel_policies = ModuleType("problems.pixel_policies")
    fake_pixel_policies.CNNMLPPolicyFactory = lambda sizes: ("cnn-mlp", tuple(sizes))
    fake_pixel_policies.AtariAgent57LiteFactory = lambda **kwargs: ("a57", kwargs)
    fake_pixel_policies.AtariCNNPolicyFactory = lambda sizes, **kwargs: ("acnn", tuple(sizes), kwargs)
    fake_pixel_policies.AtariGaussianPolicyFactory = lambda **kwargs: ("agauss", kwargs)

    monkeypatch.setitem(sys.modules, "problems.atari_env", fake_atari_env)
    monkeypatch.setitem(sys.modules, "problems.dm_control_env", fake_dm_env)
    monkeypatch.setitem(sys.modules, "problems.pixel_policies", fake_pixel_policies)

    bindings = env_conf_backends.load_atari_dm_bindings()
    assert isinstance(bindings.make_atari_preprocess_options(), _FakeAtariPreprocessOptions)

    dm_env, dm_policy = bindings.resolve_dm_control_from_tag("dm:quadruped-run:gauss", use_pixels=False)
    assert dm_env == "dm_control/quadruped-run-v0"
    assert dm_policy.__class__.__name__ == "GaussianActorBackbonePolicyFactory"

    _env, a57_policy = bindings.resolve_atari_from_tag("atari:Pong:agent57")
    assert a57_policy[0] == "a57"

    _env, mlp16_policy = bindings.resolve_atari_from_tag("atari:Pong:mlp16")
    assert mlp16_policy.__name__ == "AtariMLP16DiscretePolicy"

    assert bindings.make_dm_control_env("dm_control/cheetah-run-v0")[0] == "dm-make"
    assert bindings.make_atari_env("ALE/Pong-v5")[0] == "atari-make"


def test_register_with_env_conf_registers_both_env_conf_and_environment_spec(monkeypatch):
    from problems import env_conf_backends

    calls: list[str] = []

    monkeypatch.setattr(
        "problems.env_conf.register_atari_dm_bindings_loader",
        lambda loader: calls.append(f"env_conf:{loader.__name__}"),
    )
    monkeypatch.setattr(
        "problems.environment_spec.register_atari_dm_bindings_loader",
        lambda loader: calls.append(f"environment_spec:{loader.__name__}"),
    )

    env_conf_backends.register_with_env_conf()

    assert calls == [
        "env_conf:load_atari_dm_bindings",
        "environment_spec:load_atari_dm_bindings",
    ]
