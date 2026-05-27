from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from llm.model_client import AdapterRef, SampleBatch, SampleCall


@dataclass(frozen=True)
class NLLScoringItem:
    full_prompt: str
    target_token_ids: tuple[int, ...]
    target_start: int


def build_nll_calls(
    *,
    tokenizer: Any,
    prompts: list[str],
    answers: list[Any],
    task_obj: Any,
    lora_request_specs: list[tuple[str, int, str]] | None,
    seed: int,
) -> tuple[list[SampleCall], list[NLLScoringItem]]:
    if len(prompts) != len(answers):
        raise ValueError("prompts and answers must have the same length.")
    specs = lora_request_specs or [None for _ in prompts]
    calls: list[SampleCall] = []
    items: list[NLLScoringItem] = []
    for prompt, answer, lora_spec in zip(prompts, answers, specs, strict=True):
        item = nll_scoring_item(
            tokenizer=tokenizer,
            prompt=str(prompt),
            target_text=target_text(task_obj, answer),
        )
        calls.append(
            SampleCall(
                prompt=item.full_prompt,
                sampling=nll_sampling_kwargs(seed=seed),
                adapter=AdapterRef.from_tuple(lora_spec),
            )
        )
        items.append(item)
    return calls, items


def nll_scoring_item(*, tokenizer: Any, prompt: str, target_text: str) -> NLLScoringItem:
    prompt_ids = tuple(int(x) for x in tokenizer.encode(prompt, add_special_tokens=False))
    full_prompt = prompt + target_text
    full_ids = tuple(int(x) for x in tokenizer.encode(full_prompt, add_special_tokens=False))
    if len(full_ids) <= len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("NLL scoring requires stable prompt+target tokenization; include a target boundary such as a leading space.")
    target_token_ids = full_ids[len(prompt_ids) :]
    return NLLScoringItem(
        full_prompt=full_prompt,
        target_token_ids=target_token_ids,
        target_start=len(prompt_ids),
    )


def nll_sampling_kwargs(*, seed: int) -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "seed": int(seed),
        "max_tokens": 1,
        "n": 1,
        "prompt_logprobs": 1,
    }


def score_nll_responses(responses: list[SampleBatch], items: list[NLLScoringItem]) -> tuple[list[float], dict[str, float], list[str]]:
    if len(responses) != len(items):
        raise ValueError("responses and NLL items must have the same length.")
    fitnesses = [_score_one_nll(response.raw, item) for response, item in zip(responses, items, strict=True)]
    target_counts = [len(item.target_token_ids) for item in items]
    info = {
        "mean_nll": float(-np.mean(fitnesses)) if fitnesses else 0.0,
        "mean_target_tokens": float(np.mean(target_counts)) if target_counts else 0.0,
    }
    logs = [f"NLL: mean_nll={info['mean_nll']:.4f} mean_target_tokens={info['mean_target_tokens']:.1f}"]
    return fitnesses, info, logs


def target_text(task_obj: Any, answer: Any) -> str:
    make_target = getattr(task_obj, "target_text", None)
    if callable(make_target):
        return str(make_target(answer))
    if isinstance(answer, (str, int, float)):
        return " " + str(answer).strip()
    raise ValueError(f"{type(task_obj).__name__} does not expose target_text(answer) for NLL scoring.")


def _score_one_nll(output: Any, item: NLLScoringItem) -> float:
    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    if prompt_logprobs is None:
        raise RuntimeError("vLLM did not return prompt_logprobs; NLL scoring requires prompt_logprobs=1.")
    values = []
    for offset, token_id in enumerate(item.target_token_ids):
        index = int(item.target_start) + offset
        entry = prompt_logprobs[index] if index < len(prompt_logprobs) else None
        if entry is None:
            raise RuntimeError(f"Missing prompt logprob for target token position {index}.")
        value = entry.get(int(token_id), entry.get(str(int(token_id))))
        if value is None:
            raise RuntimeError(f"Missing prompt logprob for target token id {int(token_id)} at position {index}.")
        values.append(float(getattr(value, "logprob", value)))
    return float(np.mean(values)) if values else 0.0


__all__ = [
    "NLLScoringItem",
    "build_nll_calls",
    "nll_sampling_kwargs",
    "nll_scoring_item",
    "score_nll_responses",
    "target_text",
]
