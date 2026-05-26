from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest


def test_gymnasium_tag_parse_and_registry_surface() -> None:
    from problems.jax_env_core import supported_jax_env_tags, supports_jax_env_tag
    from problems.mjx_env import is_gymnasium_env_tag, parse_gymnasium_env_id

    assert parse_gymnasium_env_id("gymnasium:HalfCheetah-v5") == "HalfCheetah-v5"
    assert is_gymnasium_env_tag("gymnasium:Ant-v5")
    assert supports_jax_env_tag("gymnasium:HalfCheetah-v5")
    assert "gymnasium:HalfCheetah-v5" in supported_jax_env_tags()
    assert "mjx:xml:/path/to/model.xml" not in supported_jax_env_tags()


def test_gymnasium_tag_parse_rejects_empty_id() -> None:
    from problems.mjx_env import parse_gymnasium_env_id

    with pytest.raises(ValueError, match="missing an env id"):
        parse_gymnasium_env_id("gymnasium:")


def test_jax_env_factory_routes_gymnasium_to_mjx(monkeypatch) -> None:
    import problems.jax_env_factory as factory

    calls = []

    class FakeGymnasiumMJXAdapter:
        def __init__(self, env_name, *, jax, jnp):
            calls.append((env_name, jax, jnp))

    monkeypatch.setattr(factory, "GymnasiumMJXAdapter", FakeGymnasiumMJXAdapter)

    adapter = factory.make_jax_env_adapter("gymnasium:HalfCheetah-v5", jax="jax", jnp="jnp")

    assert isinstance(adapter, FakeGymnasiumMJXAdapter)
    assert calls == [("gymnasium:HalfCheetah-v5", "jax", "jnp")]


def test_mjx_adapter_loads_gymnasium_model(monkeypatch) -> None:
    import problems.mjx_env as mjx_env

    closed = []
    model = SimpleNamespace()
    env = SimpleNamespace(unwrapped=SimpleNamespace(model=model), close=lambda: closed.append(True))

    gymnasium = ModuleType("gymnasium")
    gymnasium.make = lambda _env_id: env
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)

    assert mjx_env._load_gymnasium_model("HalfCheetah-v5") is model
    assert closed == [True]
