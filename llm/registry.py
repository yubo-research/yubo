from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMEnvSpec:
    env_tag: str
    task_name: str
    task_kind: str
    dataset_name: str | None = None
    answer_format: str = "none"


@dataclass(frozen=True)
class LLMPolicySpec:
    policy_tag: str
    model_name: str
    lora_rank: int
    lora_alpha: int
    tensor_parallel_size: int = 1


_MATH_DATASETS = frozenset(
    {
        "gsm8k",
        "asdiv2k",
        "math12k",
        "orz57k",
        "deepscaler40k",
    }
)

_STATIC_ENVS: dict[str, LLMEnvSpec] = {
    "llm:zeros": LLMEnvSpec(
        env_tag="llm:zeros",
        task_name="zeros",
        task_kind="zeros",
    ),
    "llm:random": LLMEnvSpec(
        env_tag="llm:random",
        task_name="random",
        task_kind="random",
    ),
    "llm:random-boxed": LLMEnvSpec(
        env_tag="llm:random-boxed",
        task_name="random-boxed",
        task_kind="random",
        answer_format="boxed",
    ),
    "llm:countdown": LLMEnvSpec(
        env_tag="llm:countdown",
        task_name="countdown",
        task_kind="countdown",
        answer_format="answer_tags",
    ),
}

_QWEN_POLICY_RE = re.compile(r"^qwen3-(?P<size>1p7b|4b|8b|30b|32b)(?P<base>-base)?-lora-r(?P<rank>[1-9][0-9]*)$")
_QWEN_SIZE_TO_MODEL = {
    "1p7b": "1.7B",
    "4b": "4B",
    "8b": "8B",
    "30b": "30B",
    "32b": "32B",
}
_QWEN_TP_DEFAULTS = {
    "1p7b": 1,
    "4b": 1,
    "8b": 1,
    "30b": 2,
    "32b": 4,
}


def resolve_llm_env(env_tag: str) -> LLMEnvSpec:
    tag = str(env_tag)
    if tag in _STATIC_ENVS:
        return _STATIC_ENVS[tag]

    answer_tags_prefix = "llm:math:answer-tags:"
    if tag.startswith(answer_tags_prefix):
        dataset_name = tag[len(answer_tags_prefix) :]
        _validate_math_dataset(dataset_name, env_tag=tag)
        return LLMEnvSpec(
            env_tag=tag,
            task_name=f"math:answer-tags:{dataset_name}",
            task_kind="math",
            dataset_name=dataset_name,
            answer_format="answer_tags",
        )

    math_prefix = "llm:math:"
    if tag.startswith(math_prefix):
        dataset_name = tag[len(math_prefix) :]
        _validate_math_dataset(dataset_name, env_tag=tag)
        return LLMEnvSpec(
            env_tag=tag,
            task_name=f"math:{dataset_name}",
            task_kind="math",
            dataset_name=dataset_name,
            answer_format="none",
        )

    raise KeyError(f"Unknown LLM env_tag '{env_tag}'. Available examples: {supported_llm_env_tags()[:8]}")


def supported_llm_env_tags() -> tuple[str, ...]:
    math_tags = [f"llm:math:{name}" for name in sorted(_MATH_DATASETS)]
    answer_tag_math = [f"llm:math:answer-tags:{name}" for name in sorted(_MATH_DATASETS)]
    return tuple(sorted([*_STATIC_ENVS, *math_tags, *answer_tag_math]))


def resolve_llm_policy(policy_tag: str) -> LLMPolicySpec:
    tag = str(policy_tag)
    match = _QWEN_POLICY_RE.match(tag)
    if match is None:
        raise KeyError(f"Unknown LLM policy_tag '{policy_tag}'. Available examples: {supported_llm_policy_tags()[:8]}")

    size = match.group("size")
    rank = int(match.group("rank"))
    model_size = _QWEN_SIZE_TO_MODEL[size]
    suffix = f"{model_size}-Base" if match.group("base") else model_size
    return LLMPolicySpec(
        policy_tag=tag,
        model_name=f"Qwen/Qwen3-{suffix}",
        lora_rank=rank,
        lora_alpha=rank,
        tensor_parallel_size=_QWEN_TP_DEFAULTS[size],
    )


def supported_llm_policy_tags() -> tuple[str, ...]:
    tags: list[str] = []
    for size in sorted(_QWEN_SIZE_TO_MODEL):
        for rank in (1, 4):
            tags.append(f"qwen3-{size}-lora-r{rank}")
            tags.append(f"qwen3-{size}-base-lora-r{rank}")
    return tuple(sorted(tags))


def _validate_math_dataset(dataset_name: str, *, env_tag: str) -> None:
    if dataset_name not in _MATH_DATASETS:
        raise KeyError(f"Unknown math dataset in env_tag '{env_tag}'. Supported datasets: {sorted(_MATH_DATASETS)}")
