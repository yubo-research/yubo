from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from problems import atari_env, env_conf


def test_resolve_dm_control_policy_class_variants(monkeypatch):
    monkeypatch.setattr(env_conf, "_rl_gaussian", lambda variant: f"gauss:{variant}")
    monkeypatch.setattr(
        env_conf,
        "_pixel_policies",
        lambda: SimpleNamespace(CNNMLPPolicyFactory=lambda sizes: ("cnn", tuple(sizes))),
    )

    px = env_conf.get_env_conf("dm:walker-walk:pixels")
    assert px.policy_class == ("cnn", (32, 16))

    gauss = env_conf.get_env_conf("dm:walker-walk:gauss", obs_mode="vector")
    assert gauss.policy_class == "gauss:rl-gauss-tanh"

    rl_gauss = env_conf.get_env_conf("dm:walker-walk:rl-gauss", obs_mode="vector")
    assert rl_gauss.policy_class == "gauss:rl-gauss"

    default = env_conf.get_env_conf("dm:walker-walk", obs_mode="vector")
    assert default.policy_class.__class__.__name__ == "MLPPolicyFactory"


def test_resolve_atari_policy_class_variants(monkeypatch):
    fake_policy_backbone = ModuleType("rl.policy_backbone")

    class _FakeMLP16:
        pass

    fake_policy_backbone.AtariMLP16DiscretePolicy = _FakeMLP16
    monkeypatch.setitem(sys.modules, "rl.policy_backbone", fake_policy_backbone)

    assert atari_env.policy_class(policy_variant="mlp16") is _FakeMLP16

    with pytest.raises(ValueError, match="atari_agent57_factory is required"):
        atari_env.policy_class(policy_variant="agent57")

    with pytest.raises(ValueError, match="atari_gaussian_policy_factory is required"):
        atari_env.policy_class(policy_variant="gauss")

    with pytest.raises(ValueError, match="atari_cnn_policy_factory is required"):
        atari_env.policy_class(policy_variant=None)

    agent57 = atari_env.policy_class(
        policy_variant="agent57",
        atari_agent57_factory=lambda **kwargs: ("agent57", kwargs),
    )
    assert agent57[0] == "agent57"
    assert agent57[1]["cnn_variant"] == "small"

    gauss = atari_env.policy_class(
        policy_variant="gauss",
        atari_gaussian_policy_factory=lambda **kwargs: ("gauss", kwargs),
    )
    assert gauss[0] == "gauss"
    assert gauss[1]["variant"] == "small"

    default = atari_env.policy_class(
        policy_variant=None,
        atari_cnn_policy_factory=lambda sizes, **kwargs: ("cnn", tuple(sizes), kwargs),
    )
    assert default[0] == "cnn"
    assert default[1] == (24,)
