from __future__ import annotations

import re
from typing import Any, ClassVar

import numpy as np

from llm.task_modes import TaskMode


class BatchScoringTaskMixin:
    execution_mode: ClassVar[TaskMode] = TaskMode.SCORE


class RolloutTaskMixin:
    execution_mode: ClassVar[TaskMode] = TaskMode.ROLLOUT


def task_mode(task: Any) -> TaskMode:
    mode = getattr(task, "execution_mode", None)
    if isinstance(mode, TaskMode):
        return mode
    if isinstance(mode, str):
        try:
            return TaskMode(mode)
        except ValueError:
            pass
    raise TypeError(f"{type(task).__name__} must declare execution_mode as {TaskMode.SCORE.value!r} or {TaskMode.ROLLOUT.value!r}.")


def is_rollout_task(task: Any) -> bool:
    return task_mode(task) is TaskMode.ROLLOUT


def extract_model_answer(text: str, answer_format: str = "none") -> tuple[str | None, str]:
    regex_pattern = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
    regexes_to_ignore = (",", r"\$", r"(?s).*#### ", r"\.$")

    if answer_format == "none":
        answer, message = _extract_regex_answer(text, regex_pattern, last=True)
    elif answer_format == "boxed":
        answer, message = _extract_boxed_regex_answer(text, regex_pattern)
    elif answer_format == "answer_tags":
        answer, message = _extract_tagged_answer(text)
    else:
        raise ValueError(f"Unknown answer_format: {answer_format}")

    if answer is None:
        return None, message
    for pattern in regexes_to_ignore:
        answer = re.sub(pattern, "", answer)
    return answer.strip(), "answer extracted"


def score_generations(
    task: Any,
    generations: list[str],
    truncateds: list[bool],
    answer: Any,
    *,
    pass_at_k: bool = False,
) -> tuple[float, tuple[Any, ...], np.ndarray]:
    if len(generations) == 0:
        return 0.0, (), np.asarray([], dtype=np.float64)
    if len(generations) != len(truncateds):
        raise ValueError("generations and truncateds must have the same length.")

    scored = [task.score_single(generation, truncated, answer) for generation, truncated in zip(generations, truncateds, strict=True)]
    fitnesses, model_answers = zip(*scored, strict=True)
    sample_fitnesses = np.asarray(fitnesses, dtype=np.float64)
    fitness = float(np.max(sample_fitnesses) if pass_at_k else np.mean(sample_fitnesses))
    return fitness, tuple(model_answers), sample_fitnesses


def _extract_regex_answer(text: str, regex_pattern: str, *, last: bool) -> tuple[str | None, str]:
    matches = re.findall(regex_pattern, text)
    if not matches:
        return None, "No regex match found"
    match = matches[-1] if last else matches[0]
    return _first_nonempty_match(match), "answer extracted"


def _extract_boxed_regex_answer(text: str, regex_pattern: str) -> tuple[str | None, str]:
    splits = text.split("boxed{")
    if len(splits) < 2:
        return None, "No `boxed{` found"
    return _extract_regex_answer(splits[-1].strip(), regex_pattern, last=False)


def _extract_tagged_answer(text: str) -> tuple[str | None, str]:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match is None:
        return None, "No `<answer>` tags found"
    return match.group(1).strip(), "answer extracted"


def _first_nonempty_match(match: str | tuple[str, ...]) -> str:
    if isinstance(match, tuple):
        for item in match:
            if item:
                return item.strip()
        return ""
    return str(match).strip()
