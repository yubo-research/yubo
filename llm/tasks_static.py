from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm.tasks_base import extract_model_answer, score_generations


@dataclass
class ZerosTask:
    batch_size: int
    max_tokens: int

    def get_batch(self) -> tuple[list[str], list[None]]:
        prompts = ("Hello, my name is", "Write some random numbers:", "Output 3 numbers and then stop:")
        return [prompts[i % len(prompts)] for i in range(self.batch_size)], [None for _ in range(self.batch_size)]

    def score(
        self,
        generations: list[str],
        truncateds: list[bool],
        answer: Any,
        *,
        pass_at_k: bool = False,
    ):
        return score_generations(self, generations, truncateds, answer, pass_at_k=pass_at_k)

    def score_single(self, generation: str, truncated: bool, answer: Any) -> tuple[float, None]:
        _ = answer
        if truncated:
            return 0.0, None
        return float(sum(c == "0" for c in generation)) / float(max(self.max_tokens, 1)), None


class RandomTask:
    def __init__(self, *, batch_size: int, max_random_number: int, seed: int, answer_format: str = "none") -> None:
        import numpy as np

        self.batch_size = int(batch_size)
        self.max_random_number = int(max_random_number)
        self.answer_format = str(answer_format)
        self.rng = np.random.default_rng(int(seed))
        self.prompt = self._make_prompt()

    def get_batch(self) -> tuple[list[str], list[int]]:
        answers = self.rng.integers(1, self.max_random_number + 1, size=self.batch_size).tolist()
        return [self.prompt for _ in range(self.batch_size)], answers

    def score(
        self,
        generations: list[str],
        truncateds: list[bool],
        answer: Any,
        *,
        pass_at_k: bool = False,
    ):
        return score_generations(self, generations, truncateds, answer, pass_at_k=pass_at_k)

    def score_single(self, generation: str, truncated: bool, answer: Any) -> tuple[float, int | None]:
        if truncated:
            return 0.0, None
        model_answer = self._parse_answer(generation)
        return (1.0 if model_answer == int(answer) else 0.0), model_answer

    def _make_prompt(self) -> str:
        prompt = f"Pick a random number between 1 and {self.max_random_number} (inclusive)."
        if self.answer_format == "boxed":
            prompt += " Format your pick in \\boxed{}."
        elif self.answer_format != "none":
            raise ValueError(f"Unknown answer_format for random task: {self.answer_format}")
        return f"User: {prompt}\n\nAssistant:"

    def _parse_answer(self, generation: str) -> int | None:
        model_answer_raw, _ = extract_model_answer(generation, answer_format=self.answer_format)
        try:
            return int(model_answer_raw) if model_answer_raw is not None else None
        except ValueError:
            return None
