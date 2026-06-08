from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from llm.model_client import SampleBatch


@dataclass
class ScoreAccumulator:
    num_prompts: int
    fitness_list: list[float] = field(default_factory=list)
    distinct_counts: list[int] = field(default_factory=list)
    mean_char_lengths: list[float] = field(default_factory=list)
    mean_token_lengths: list[float] = field(default_factory=list)
    responses_for_logging: list[str] = field(default_factory=list)
    sample_stds: list[float] = field(default_factory=list)
    pass_at_k_fitnesses: list[float] = field(default_factory=list)
    mean_fitnesses: list[float] = field(default_factory=list)
    total_responses: int = 0
    num_truncated: int = 0
    pop_responses_buffer: str = ""

    def add(self, scored: "ScoredOutput") -> None:
        self.fitness_list.append(float(scored.fit))
        self.distinct_counts.append(scored.distinct_count)
        self.mean_char_lengths.append(float(np.mean(scored.char_lengths)))
        self.mean_token_lengths.append(float(np.mean(scored.token_lengths)))
        self.total_responses += len(scored.char_lengths)
        self.num_truncated += scored.num_truncated
        self._add_sample_fitnesses(scored)
        self._add_log(scored)

    def _add_sample_fitnesses(self, scored: "ScoredOutput") -> None:
        if len(scored.sample_fitnesses) == 0:
            self.pass_at_k_fitnesses.append(float(scored.fit))
            self.mean_fitnesses.append(float(scored.fit))
            return
        self.pass_at_k_fitnesses.append(float(np.max(scored.sample_fitnesses)))
        self.mean_fitnesses.append(float(np.mean(scored.sample_fitnesses)))
        if len(scored.sample_fitnesses) > 1:
            self.sample_stds.append(float(np.std(scored.sample_fitnesses)))

    def _add_log(self, scored: "ScoredOutput") -> None:
        if scored.prompt_log:
            self.pop_responses_buffer += scored.prompt_log
        if scored.is_population_boundary and self.pop_responses_buffer:
            self.responses_for_logging.append(f"-----POP {scored.pop_idx} BATCH LOG-----\n{self.pop_responses_buffer}")
            self.pop_responses_buffer = ""

    def final(self) -> tuple[list[float], dict[str, float], list[str]]:
        info = {
            "total_responses": float(self.total_responses),
            "prop_truncated": float(self.num_truncated / self.total_responses) if self.total_responses else 0.0,
            "mean_char_length": _mean_or_zero(self.mean_char_lengths),
            "mean_token_length": _mean_or_zero(self.mean_token_lengths),
            "mean_distinct_counts": _mean_or_zero(self.distinct_counts),
            "std_in_samples": _mean_or_zero(self.sample_stds),
            "pass_at_k_fitness": _mean_or_zero(self.pass_at_k_fitnesses),
            "mean_sample_fitness": _mean_or_zero(self.mean_fitnesses),
        }
        return self.fitness_list, info, self.responses_for_logging


@dataclass(frozen=True)
class ScoredOutput:
    fit: float
    sample_fitnesses: np.ndarray
    distinct_count: int
    char_lengths: list[int]
    token_lengths: list[int]
    num_truncated: int
    prompt_log: str
    pop_idx: int
    is_population_boundary: bool


def score_request_outputs(
    request_outputs: list[Any],
    *,
    prompts: list[str],
    task_obj: Any,
    answers: list[Any],
    pass_at_k: bool,
) -> tuple[list[float], dict[str, float], list[str]]:
    accumulator = ScoreAccumulator(num_prompts=len(answers))
    for index, output in enumerate(request_outputs):
        accumulator.add(
            _score_one_output(
                index,
                output,
                prompts=prompts,
                task_obj=task_obj,
                answers=answers,
                pass_at_k=pass_at_k,
            )
        )
    return accumulator.final()


def score_completions(
    responses: list[SampleBatch],
    *,
    prompts: list[str],
    task_obj: Any,
    answers: list[Any],
    pass_at_k: bool,
) -> tuple[list[float], dict[str, float], list[str]]:
    request_outputs = [SimpleNamespace(outputs=response.completions) for response in responses]
    return score_request_outputs(
        request_outputs,
        prompts=prompts,
        task_obj=task_obj,
        answers=answers,
        pass_at_k=pass_at_k,
    )


def _score_one_output(
    index: int,
    output: Any,
    *,
    prompts: list[str],
    task_obj: Any,
    answers: list[Any],
    pass_at_k: bool,
) -> ScoredOutput:
    num_prompts = len(answers)
    prompt_idx = index % num_prompts
    pop_idx = index // num_prompts
    responses = [sample.text for sample in output.outputs]
    truncateds = [sample.finish_reason == "length" for sample in output.outputs]
    fit, model_answers, sample_fitnesses = task_obj.score(responses, truncateds, answers[prompt_idx], pass_at_k=pass_at_k)
    sample_fitnesses = np.asarray(sample_fitnesses, dtype=np.float64)
    prompt_log = _prompt_log(
        pop_idx,
        prompt_idx,
        prompt=prompts[index],
        samples=output.outputs,
        sample_fitnesses=sample_fitnesses,
        fit=float(fit),
    )
    char_lengths, token_lengths, num_truncated = _sample_lengths(output.outputs)
    return ScoredOutput(
        fit=float(fit),
        sample_fitnesses=sample_fitnesses,
        distinct_count=_distinct_answer_count(model_answers),
        char_lengths=char_lengths,
        token_lengths=token_lengths,
        num_truncated=num_truncated,
        prompt_log=prompt_log,
        pop_idx=pop_idx,
        is_population_boundary=(index + 1) % num_prompts == 0,
    )


def _prompt_log(
    pop_idx: int,
    prompt_idx: int,
    *,
    prompt: str,
    samples: list[Any],
    sample_fitnesses: np.ndarray,
    fit: float,
) -> str:
    if pop_idx >= 2 or prompt_idx >= 3:
        return ""
    chunks = [f"\n[PROMPT {prompt_idx}]: {prompt}\n"]
    for sample_idx, sample in enumerate(samples):
        sample_fit = float(sample_fitnesses[sample_idx]) if sample_idx < len(sample_fitnesses) else float(fit)
        chunks.append(f"\n------SAMPLE {sample_idx + 1}: {sample.text} || FIT={sample_fit}\n")
    return "".join(chunks)


def _sample_lengths(samples: list[Any]) -> tuple[list[int], list[int], int]:
    char_lengths = [len(sample.text) for sample in samples]
    token_lengths = [len(sample.token_ids) for sample in samples]
    num_truncated = sum(1 for sample in samples if sample.finish_reason == "length")
    return char_lengths, token_lengths, num_truncated


def _distinct_answer_count(model_answers: list[Any]) -> int:
    keys = {_answer_key(answer) for answer in model_answers if answer is not None}
    return len(keys)


def _answer_key(answer: Any) -> Any:
    return tuple(answer) if isinstance(answer, list) else answer


def _mean_or_zero(values: list[Any]) -> float:
    return float(np.mean(values)) if values else 0.0
