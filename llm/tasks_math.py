from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from llm.tasks_base import (
    BatchScoringTaskMixin,
    extract_model_answer,
    score_generations,
)


@dataclass(frozen=True)
class MathTaskConfig:
    batch_size: int
    dataset_name: str
    seed: int = 0
    dataset_size: int | None = None
    answer_format: str = "none"
    apply_chat_template: bool = False
    tokenizer: Any | None = None

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> MathTaskConfig:
        return cls(**kwargs)


class MathTask(BatchScoringTaskMixin):
    def __init__(self, config: MathTaskConfig | None = None, **kwargs: Any) -> None:
        cfg = config if config is not None else MathTaskConfig.from_kwargs(**kwargs)
        self.batch_size = int(cfg.batch_size)
        self.dataset_name = str(cfg.dataset_name).lower()
        self.answer_format = str(cfg.answer_format)
        self.apply_chat_template = bool(cfg.apply_chat_template)
        self.tokenizer = cfg.tokenizer
        self.idx = 0
        self.dataset, self.is_train, self.split_names = _load_math_dataset(
            self.dataset_name,
            seed=int(cfg.seed),
            dataset_size=cfg.dataset_size,
        )

    def get_batch(self) -> tuple[list[str], list[str]]:
        if not self.is_train:
            raise ValueError(f"get_batch requires a train dataset, got {self.dataset_name!r}.")
        indices = np.arange(self.idx, self.idx + self.batch_size) % len(self.dataset)
        self.idx += self.batch_size
        examples = [self.dataset[int(i)] for i in indices]
        self._last_examples = examples
        return self._format_examples(examples)

    def nll_user_contents(self, prompts: list[str], answers: list[Any]) -> list[str]:
        if not self.apply_chat_template:
            return [None for _ in prompts]
        examples = getattr(self, "_last_examples", None)
        if examples is None or len(examples) != len(prompts):
            raise RuntimeError("MathTask.nll_user_contents requires a fresh get_batch() call for this prompt batch.")
        return [f"{example['problem']}\n{_math_instruction(self.answer_format)}" for example in examples]

    def get_eval_batch(self) -> tuple[list[str], list[str]]:
        if self.is_train:
            raise ValueError(f"get_eval_batch requires an eval dataset, got {self.dataset_name!r}.")
        examples = []
        for split in self.split_names:
            split_dataset = self.dataset[split]
            examples.extend(split_dataset[int(i % len(split_dataset))] for i in range(self.batch_size))
        return self._format_examples(examples)

    def state_dict(self) -> dict[str, int]:
        return {"idx": int(self.idx)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.idx = int(state.get("idx", 0))

    def score(
        self,
        generations: list[str],
        truncateds: list[bool],
        answer: Any,
        *,
        pass_at_k: bool = False,
    ):
        return score_generations(self, generations, truncateds, answer, pass_at_k=pass_at_k)

    def score_single(self, generation: str, truncated: bool, answer: Any) -> tuple[float, str | None]:
        if truncated and not _has_explicit_math_answer(generation, answer_format=self.answer_format):
            return 0.0, None
        is_correct, model_answer = check_math_correct(generation, answer, answer_format=self.answer_format)
        return (1.0 if is_correct else 0.0), model_answer

    def target_text(self, answer: Any) -> str:
        value = _math_target_answer(answer)
        if self.answer_format == "answer_tags":
            return f" <answer>{value}</answer>"
        return f" \\boxed{{{value}}}"

    def _format_examples(self, examples: list[Any]) -> tuple[list[str], list[str]]:
        prompts = [self._format_prompt(example) for example in examples]
        answers = [example["answer"] for example in examples]
        return prompts, answers

    def _format_prompt(self, example: Any) -> str:
        problem = f"{example['problem']}\n{_math_instruction(self.answer_format)}"
        if self.apply_chat_template:
            if self.tokenizer is None:
                raise ValueError("apply_chat_template=True requires a tokenizer.")
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": problem}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"User: {problem}\nAssistant: <think>"


def _math_target_answer(answer: Any) -> str:
    value = str(answer[0] if isinstance(answer, list) else answer)
    if "####" in value:
        value = value.rsplit("####", 1)[1]
    return value.strip()


def _load_math_dataset(dataset_name: str, *, seed: int, dataset_size: int | None) -> tuple[Any, bool, list[str]]:
    dataset_names = {
        "gsm8k": ("axon-rl/GSM-8k", "train", True),
        "asdiv2k": ("axon-rl/ASDIV-2k", "train", True),
        "math12k": ("axon-rl/MATH-12k", "train", True),
        "orz57k": ("axon-rl/ORZ-57k", "train", True),
        "deepscaler40k": ("axon-rl/DeepScaleR-40K", "train", True),
        "math-eval": (
            "axon-rl/math-eval",
            ["math", "amc", "olympiad_bench", "minerva", "aime24"],
            False,
        ),
    }
    if dataset_name not in dataset_names:
        raise ValueError(f"Unknown math dataset {dataset_name!r}. Supported: {sorted(dataset_names)}")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("LLM math tasks require the `datasets` package.") from exc

    hf_name, splits, is_train = dataset_names[dataset_name]
    if is_train:
        return _load_train_math_dataset(load_dataset, hf_name, str(splits), seed=seed, dataset_size=dataset_size)
    return _load_eval_math_dataset(load_dataset, hf_name, list(splits), seed=seed, dataset_size=dataset_size)


def check_math_correct(generation: str, answer: Any, *, answer_format: str = "none") -> tuple[bool, str | None]:
    correct_answers = [str(x) for x in answer] if isinstance(answer, list) else [str(answer)]
    model_answer = _extract_math_answer(generation, answer_format=answer_format)
    if model_answer is None:
        return False, None
    return any(_grade_math_answer(model_answer, correct_answer) for correct_answer in correct_answers), model_answer


def _extract_math_answer(generation: str, *, answer_format: str) -> str | None:
    if answer_format == "answer_tags":
        model_answer, _ = extract_model_answer(generation, answer_format="answer_tags")
        return model_answer
    boxed_answer = _last_boxed_answer(generation)
    if boxed_answer is not None:
        return boxed_answer
    model_answer, _ = extract_model_answer(generation, answer_format="none")
    return model_answer


def _has_explicit_math_answer(generation: str, *, answer_format: str) -> bool:
    if answer_format == "answer_tags":
        model_answer, _ = extract_model_answer(generation, answer_format="answer_tags")
        return model_answer is not None
    return _last_boxed_answer(generation) is not None


def _grade_math_answer(model_answer: str, correct_answer: str) -> bool:
    if _normalize_answer(model_answer) == _normalize_answer(correct_answer):
        return True
    return _grade_math_answer_with_math_verify(model_answer, correct_answer)


def _normalize_answer(value: str) -> str:
    return re.sub(r"\s+", "", str(value).strip().replace(",", "").replace("$", ""))


def _last_boxed_answer(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    return _balanced_brace_content(text, start + len(marker))


def _grade_math_answer_with_math_verify(model_answer: str, correct_answer: str) -> bool:
    try:
        from math_verify import (
            ExprExtractionConfig,
            LatexExtractionConfig,
            parse,
            verify,
        )
    except ImportError:
        return False

    extraction_config = (LatexExtractionConfig(), ExprExtractionConfig())
    try:
        golds = parse(correct_answer, extraction_config, parsing_timeout=2)
        preds = parse(model_answer, extraction_config, parsing_timeout=2)
    except Exception:
        return False
    if not golds or not preds:
        return False
    try:
        return any(bool(verify(gold, pred, timeout_seconds=2)) for gold in golds for pred in preds)
    except Exception:
        return False


def _load_train_math_dataset(load_dataset: Any, hf_name: str, split: str, *, seed: int, dataset_size: int | None):
    dataset = load_dataset(hf_name, split=split).shuffle(seed=seed)
    if dataset_size is not None:
        dataset = dataset.select(range(min(int(dataset_size), len(dataset))))
    return dataset, True, []


def _load_eval_math_dataset(
    load_dataset: Any,
    hf_name: str,
    split_names: list[str],
    *,
    seed: int,
    dataset_size: int | None,
):
    raw = dict(load_dataset(hf_name))
    raw["gsm8k"] = load_dataset("axon-rl/GSM-8k", split="train").shuffle(seed=seed).select(range(500))
    raw["asdiv"] = load_dataset("axon-rl/ASDIV-2k", split="train").shuffle(seed=seed).select(range(500))
    raw["aime25"] = load_dataset("math-ai/aime25", split="test").shuffle(seed=seed)
    split_names.extend(["gsm8k", "asdiv", "aime25"])
    if dataset_size is not None:
        limit = int(dataset_size)
        raw = {name: data.select(range(min(limit, len(data)))) for name, data in raw.items()}
    return (
        {name: data.shuffle(seed=seed) for name, data in raw.items()},
        False,
        split_names,
    )


def _math_instruction(answer_format: str) -> str:
    if answer_format == "answer_tags":
        return "Please reason step-by-step concisely."
    return "Please reason step-by-step concisely, and put your final answer within \\boxed{ }."


def _balanced_brace_content(text: str, start: int) -> str | None:
    depth = 1
    out = []
    for char in text[start:]:
        if char == "{":
            depth += 1
        if char == "}":
            depth -= 1
        if depth == 0:
            answer = "".join(out).strip()
            return answer or None
        out.append(char)
    return None
