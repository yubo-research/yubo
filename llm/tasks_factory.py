from __future__ import annotations

from typing import Any

from llm.console_observer import UnifiedConsoleManager
from llm.registry import LLMEnvSpec
from llm.tasks_base import LLMTask
from llm.tasks_countdown import CountdownTask
from llm.tasks_math import MathTask
from llm.tasks_static import RandomTask, ZerosTask
from llm.tasks_verifiers import VerifiersTask
from llm.thm_task import TheoremProvingTask


def build_task(
    spec: LLMEnvSpec,
    *,
    batch_size: int,
    seed: int,
    max_tokens: int,
    dataset_size: int | None = None,
    tokenizer: Any | None = None,
    apply_chat_template: bool = False,
    console: UnifiedConsoleManager | None = None,
) -> LLMTask:
    if spec.task_kind == "zeros":
        return ZerosTask(batch_size=batch_size, max_tokens=max_tokens)
    if spec.task_kind == "random":
        return RandomTask(
            batch_size=batch_size,
            max_random_number=4,
            seed=seed,
            answer_format=spec.answer_format,
        )
    if spec.task_kind == "math":
        return _build_math_task(
            spec,
            batch_size=batch_size,
            seed=seed,
            dataset_size=dataset_size,
            tokenizer=tokenizer,
            apply_chat_template=apply_chat_template,
        )
    if spec.task_kind == "countdown":
        return CountdownTask(batch_size=batch_size, seed=seed, dataset_size=dataset_size)
    if spec.task_kind == "verifiers":
        if spec.dataset_name is None:
            raise ValueError(f"verifiers LLM env requires dataset_name/env_id: {spec}")
        return VerifiersTask(
            batch_size=batch_size,
            env_id=spec.dataset_name,
            seed=seed,
            dataset_size=dataset_size,
            tokenizer=tokenizer,
            apply_chat_template=apply_chat_template,
        )
    if spec.task_kind == "thm":
        # task_name format: thm:{lang}:{dataset}
        lang = spec.task_name.split(":")[1]
        return TheoremProvingTask(
            batch_size=batch_size,
            language=lang,
            dataset_name=spec.dataset_name,
            seed=seed,
            tokenizer=tokenizer,
            console=console,
        )
    raise ValueError(f"Unsupported LLM task kind: {spec.task_kind}")


def _build_math_task(
    spec: LLMEnvSpec,
    *,
    batch_size: int,
    seed: int,
    dataset_size: int | None,
    tokenizer: Any | None,
    apply_chat_template: bool,
) -> MathTask:
    if spec.dataset_name is None:
        raise ValueError(f"Math LLM env requires dataset_name: {spec}")
    return MathTask(
        batch_size=batch_size,
        dataset_name=spec.dataset_name,
        seed=seed,
        dataset_size=dataset_size,
        answer_format=spec.answer_format,
        tokenizer=tokenizer,
        apply_chat_template=apply_chat_template,
    )
