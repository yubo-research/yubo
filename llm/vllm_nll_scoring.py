from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from llm.model_client import AdapterRef, SampleBatch, SampleCall


@dataclass(frozen=True)
class NLLScoringItem:
    full_prompt: str
    target_token_ids: tuple[int, ...]
    target_start: int
    full_token_ids: tuple[int, ...]


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
    user_contents = _nll_user_contents(task_obj, prompts, answers)
    calls: list[SampleCall] = []
    items: list[NLLScoringItem] = []
    for prompt, answer, lora_spec, user_content in zip(prompts, answers, specs, user_contents, strict=True):
        target = target_text(task_obj, answer)
        if user_content is not None:
            item = nll_scoring_item_chat(
                tokenizer=tokenizer,
                prompt=str(prompt),
                target_text=target,
                user_content=str(user_content),
            )
        else:
            item = nll_scoring_item(
                tokenizer=tokenizer,
                prompt=str(prompt),
                target_text=target,
            )
        if nll_use_prefix_decode():
            for offset, token_id in enumerate(item.target_token_ids):
                prefix_end = int(item.target_start) + offset
                calls.append(
                    SampleCall(
                        prompt=_stable_prefix_text(tokenizer, item.full_token_ids, prefix_end),
                        sampling=nll_prefix_sampling_kwargs(seed=seed, token_id=int(token_id)),
                        adapter=AdapterRef.from_tuple(lora_spec),
                    )
                )
            items.append(item)
            continue
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
        full_token_ids=full_ids,
    )


def nll_scoring_item_chat(
    *,
    tokenizer: Any,
    prompt: str,
    target_text: str,
    user_content: str,
) -> NLLScoringItem:
    """Build a chat-templated full prompt so target tokens stay inside the assistant turn."""
    assistant_content = str(target_text).strip()
    full_prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": str(user_content)},
            {"role": "assistant", "content": assistant_content},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tuple(int(x) for x in tokenizer.encode(prompt, add_special_tokens=False))
    full_ids = tuple(int(x) for x in tokenizer.encode(full_prompt, add_special_tokens=False))
    if len(full_ids) <= len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("NLL chat scoring requires the user prompt prefix to match the templated full prompt; check apply_chat_template settings.")
    target_token_ids = full_ids[len(prompt_ids) :]
    return NLLScoringItem(
        full_prompt=str(full_prompt),
        target_token_ids=target_token_ids,
        target_start=len(prompt_ids),
        full_token_ids=full_ids,
    )


def nll_sampling_kwargs(*, seed: int) -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "seed": int(seed),
        "max_tokens": 1,
        "n": 1,
        "prompt_logprobs": 1,
    }


def nll_prefix_sampling_kwargs(*, seed: int, token_id: int) -> dict[str, Any]:
    """Score one target token via prefix decode (vllm-metal lacks prompt_logprobs).

    ``logprobs=1`` alone returns only the model's top-1 token, not an arbitrary
    label token. ``allowed_token_ids`` forces greedy decode onto the label so
    the sampled-token logprob matches teacher-forced NLL.
    """
    return {
        "temperature": 0.0,
        "seed": int(seed),
        "max_tokens": 1,
        "n": 1,
        "logprobs": 1,
        "allowed_token_ids": [int(token_id)],
    }


def nll_use_prefix_decode() -> bool:
    override = os.environ.get("YUBO_NLL_PREFIX_DECODE")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}
    return os.uname().sysname == "Darwin"


def score_nll_responses(responses: list[SampleBatch], items: list[NLLScoringItem]) -> tuple[list[float], dict[str, float], list[str]]:
    if len(responses) != len(items) and not nll_use_prefix_decode():
        raise ValueError("responses and NLL items must have the same length.")
    if nll_use_prefix_decode():
        fitnesses = _score_nll_prefix_responses(responses, items)
    else:
        if len(responses) != len(items):
            raise ValueError("responses and NLL items must have the same length.")
        fitnesses = [_score_one_nll(response.raw, item) for response, item in zip(responses, items, strict=True)]
    target_counts = [len(item.target_token_ids) for item in items]
    info = {
        "mean_nll": float(-np.mean(fitnesses)) if fitnesses else 0.0,
        "mean_target_tokens": float(np.mean(target_counts)) if target_counts else 0.0,
    }
    mode = "prefix" if nll_use_prefix_decode() else "prompt"
    logs = [f"NLL({mode}): mean_nll={info['mean_nll']:.4f} mean_target_tokens={info['mean_target_tokens']:.1f}"]
    return fitnesses, info, logs


def target_text(task_obj: Any, answer: Any) -> str:
    make_target = getattr(task_obj, "target_text", None)
    if callable(make_target):
        return str(make_target(answer))
    if isinstance(answer, (str, int, float)):
        return " " + str(answer).strip()
    raise ValueError(f"{type(task_obj).__name__} does not expose target_text(answer) for NLL scoring.")


def _nll_user_contents(task_obj: Any, prompts: list[str], answers: list[Any]) -> list[str | None]:
    getter = getattr(task_obj, "nll_user_contents", None)
    if callable(getter):
        return list(getter(prompts, answers))
    return [None for _ in prompts]


def _resolve_target_start(
    prompt_token_ids: tuple[int, ...] | list[int],
    target_token_ids: tuple[int, ...],
    *,
    fallback: int,
) -> int:
    ids = tuple(int(x) for x in prompt_token_ids)
    target = tuple(int(x) for x in target_token_ids)
    if not target:
        return int(fallback)
    for start in range(len(ids) - len(target), -1, -1):
        if ids[start : start + len(target)] == target:
            return start
    if 0 <= int(fallback) <= len(ids) - len(target):
        return int(fallback)
    raise RuntimeError(f"Could not align NLL target tokens with vLLM prompt_token_ids (target_len={len(target)}, prompt_len={len(ids)}, fallback={fallback}).")


def _prompt_logprob_entry(prompt_logprobs: Any, index: int) -> Any:
    if index < 0 or index >= len(prompt_logprobs):
        return None
    return prompt_logprobs[index]


def _extract_logprob_value(entry: Any, token_id: int) -> float | None:
    if entry is None:
        return None
    value = entry.get(int(token_id), entry.get(str(int(token_id))))
    if value is None:
        return None
    return float(getattr(value, "logprob", value))


def _stable_prefix_text(tokenizer: Any, full_token_ids: tuple[int, ...], end: int) -> str:
    prefix_ids = [int(x) for x in full_token_ids[:end]]
    text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    recoded = [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
    if recoded != prefix_ids:
        raise ValueError(f"NLL prefix decode round-trip failed (end={end}, expected {len(prefix_ids)} token ids, got {len(recoded)}).")
    return text


def _score_nll_prefix_responses(responses: list[SampleBatch], items: list[NLLScoringItem]) -> list[float]:
    expected = sum(len(item.target_token_ids) for item in items)
    if len(responses) != expected:
        raise ValueError(f"Expected {expected} prefix NLL responses for {len(items)} items, got {len(responses)}.")
    offset = 0
    fitnesses: list[float] = []
    for item in items:
        chunk = responses[offset : offset + len(item.target_token_ids)]
        offset += len(item.target_token_ids)
        values = [_score_one_prefix_nll(response.raw, int(token_id)) for response, token_id in zip(chunk, item.target_token_ids, strict=True)]
        fitnesses.append(float(np.mean(values)) if values else 0.0)
    return fitnesses


def _score_one_prefix_nll(output: Any, token_id: int) -> float:
    outputs = getattr(output, "outputs", None) or ()
    if not outputs:
        raise RuntimeError("vLLM prefix NLL scoring returned no outputs.")
    sample_logprobs = getattr(outputs[0], "logprobs", None)
    if sample_logprobs is None or len(sample_logprobs) == 0:
        raise RuntimeError("vLLM prefix NLL scoring returned no sample logprobs.")
    entry = sample_logprobs[0]
    value = _extract_logprob_value(entry, int(token_id))
    if value is None and isinstance(entry, dict) and len(entry) == 1:
        only_key = next(iter(entry))
        value = _extract_logprob_value(entry, only_key)
    if value is None:
        raise RuntimeError(f"Missing sample logprob for token id {int(token_id)}.")
    return value


def _score_one_nll(output: Any, item: NLLScoringItem) -> float:
    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    if prompt_logprobs is None:
        raise RuntimeError("vLLM did not return prompt_logprobs; NLL scoring requires prompt_logprobs=1.")
    prompt_token_ids = getattr(output, "prompt_token_ids", None)
    target_start = (
        _resolve_target_start(prompt_token_ids, item.target_token_ids, fallback=int(item.target_start))
        if prompt_token_ids is not None
        else int(item.target_start)
    )
    values = []
    for offset, token_id in enumerate(item.target_token_ids):
        index = int(target_start) + offset
        value = _extract_logprob_value(_prompt_logprob_entry(prompt_logprobs, index), int(token_id))
        if value is None:
            raise RuntimeError(
                f"Missing prompt logprob for target token id {int(token_id)} at position {index} "
                f"(target_start={target_start}, prompt_len={len(prompt_token_ids or ())}, "
                f"logprob_len={len(prompt_logprobs)})."
            )
        values.append(value)
    return float(np.mean(values)) if values else 0.0


__all__ = [
    "NLLScoringItem",
    "build_nll_calls",
    "nll_prefix_sampling_kwargs",
    "nll_sampling_kwargs",
    "nll_scoring_item",
    "nll_scoring_item_chat",
    "nll_use_prefix_decode",
    "score_nll_responses",
    "target_text",
]
