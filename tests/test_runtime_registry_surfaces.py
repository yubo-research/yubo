def test_jax_env_examples_are_first_class() -> None:
    from problems.jax_env_core import (
        supported_jax_env_tags,
        supports_jax_env_tag,
    )

    tags = supported_jax_env_tags()

    assert "jumanji:Game2048-v1" in tags
    assert "brax:ant" in tags
    assert "gymnasium:HalfCheetah-v5" in tags
    assert "mjx:xml:/path/to/model.xml" not in tags
    assert "synthetic:linear-speed" not in tags
    assert all(supports_jax_env_tag(tag) for tag in tags)


def test_eggroll_policy_examples_resolve() -> None:
    from policies.registry import get_policy_preset

    policy_tags = (
        "eggroll-ac-mlp-256x3-pqn",
        "eggroll-marl-mlp-64x3-tanh",
        "nanoegg:int8:6l:256d",
    )

    for policy_tag in policy_tags:
        assert get_policy_preset(policy_tag) is not None


def test_pretrain_examples_are_first_class() -> None:
    from problems.pre_obj import (
        resolve_hyperscalees_pretrain_spec,
        resolve_nanoegg_pretrain_spec,
        supported_hyperscalees_pretrain_env_tags,
        supported_nanoegg_pretrain_examples,
    )

    for env_tag in supported_hyperscalees_pretrain_env_tags():
        assert resolve_hyperscalees_pretrain_spec(env_tag).env_tag == env_tag
    for env_tag, policy_tag in supported_nanoegg_pretrain_examples():
        spec = resolve_nanoegg_pretrain_spec(env_tag, policy_tag)
        assert spec.env_tag == env_tag
        assert spec.policy_tag == policy_tag


def test_llm_registry_examples_are_first_class() -> None:
    from llm.registry import resolve_llm_env, resolve_llm_policy

    assert resolve_llm_env("llm:math:gsm8k").task_kind == "math"
    assert resolve_llm_policy("qwen3-1p7b-lora-r1").model_name == "Qwen/Qwen3-1.7B"
