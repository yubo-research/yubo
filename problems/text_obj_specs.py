from __future__ import annotations

from dataclasses import dataclass

from llm.registry import LLMEnvSpec, LLMPolicySpec, resolve_llm_env, resolve_llm_policy


_TEXT_TAG_PREFIX = "llm:"


@dataclass(frozen=True)
class TextSpec:
    env: LLMEnvSpec
    policy: LLMPolicySpec


def is_text_env(env_tag: str) -> bool:
    return str(env_tag).startswith(_TEXT_TAG_PREFIX)


def resolve_text_spec(env_tag: str, policy_tag: str | None) -> TextSpec:
    if not is_text_env(env_tag):
        raise ValueError(f"Unsupported text env_tag: {env_tag!r}. Expected prefix {_TEXT_TAG_PREFIX!r}.")
    if policy_tag is None:
        raise ValueError("UHD text objectives require policy_tag.")
    return TextSpec(env=resolve_llm_env(str(env_tag)), policy=resolve_llm_policy(str(policy_tag)))
