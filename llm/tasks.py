from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from llm.registry import LLMEnvSpec


class LLMTask(Protocol):
    def get_batch(self) -> tuple[list[str], list[Any]]: ...

    def score(self, generations: list[str], truncateds: list[bool], answer: Any, *, pass_at_k: bool = False) -> tuple[float, tuple[Any, ...], np.ndarray]: ...


def extract_model_answer(text: str, answer_format: str = "none") -> tuple[str | None, str]:
    regex_pattern = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
    regexes_to_ignore = (",", r"\$", r"(?s).*#### ", r"\.$")

    if answer_format == "none":
        matches = re.findall(regex_pattern, text)
        if not matches:
            return None, "No regex match found"
        match = matches[-1]
        answer = _first_nonempty_match(match)
    elif answer_format == "boxed":
        splits = text.split("boxed{")
        if len(splits) < 2:
            return None, "No `boxed{` found"
        matches = re.findall(regex_pattern, splits[-1].strip())
        if not matches:
            return None, "No regex match found"
        answer = _first_nonempty_match(matches[0])
    elif answer_format == "answer_tags":
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match is None:
            return None, "No `<answer>` tags found"
        answer = match.group(1).strip()
    else:
        raise ValueError(f"Unknown answer_format: {answer_format}")

    for pattern in regexes_to_ignore:
        answer = re.sub(pattern, "", answer)
    return answer.strip(), "answer extracted"


def score_generations(task: Any, generations: list[str], truncateds: list[bool], answer: Any, *, pass_at_k: bool = False) -> tuple[float, tuple[Any, ...], np.ndarray]:
    if len(generations) == 0:
        return 0.0, (), np.asarray([], dtype=np.float64)
    if len(generations) != len(truncateds):
        raise ValueError("generations and truncateds must have the same length.")

    scored = [task.score_single(generation, truncated, answer) for generation, truncated in zip(generations, truncateds, strict=True)]
    fitnesses, model_answers = zip(*scored, strict=True)
    sample_fitnesses = np.asarray(fitnesses, dtype=np.float64)
    fitness = float(np.max(sample_fitnesses) if pass_at_k else np.mean(sample_fitnesses))
    return fitness, tuple(model_answers), sample_fitnesses


@dataclass
class ZerosTask:
    batch_size: int
    max_tokens: int

    def get_batch(self) -> tuple[list[str], list[None]]:
        prompts = ("Hello, my name is", "Write some random numbers:", "Output 3 numbers and then stop:")
        return [prompts[i % len(prompts)] for i in range(self.batch_size)], [None for _ in range(self.batch_size)]

    def score(self, generations: list[str], truncateds: list[bool], answer: Any, *, pass_at_k: bool = False) -> tuple[float, tuple[Any, ...], np.ndarray]:
        return score_generations(self, generations, truncateds, answer, pass_at_k=pass_at_k)

    def score_single(self, generation: str, truncated: bool, answer: Any) -> tuple[float, None]:
        if truncated:
            return 0.0, None
        return float(sum(c == "0" for c in generation)) / float(max(self.max_tokens, 1)), None


class RandomTask:
    def __init__(self, *, batch_size: int, max_random_number: int, seed: int, answer_format: str = "none") -> None:
        self.batch_size = int(batch_size)
        self.max_random_number = int(max_random_number)
        self.answer_format = str(answer_format)
        self.rng = np.random.default_rng(int(seed))
        prompt = f"Pick a random number between 1 and {self.max_random_number} (inclusive)."
        if self.answer_format == "boxed":
            prompt += " Format your pick in \\boxed{}."
        elif self.answer_format != "none":
            raise ValueError(f"Unknown answer_format for random task: {self.answer_format}")
        self.prompt = f"User: {prompt}\n\nAssistant:"

    def get_batch(self) -> tuple[list[str], list[int]]:
        answers = self.rng.integers(1, self.max_random_number + 1, size=self.batch_size).tolist()
        return [self.prompt for _ in range(self.batch_size)], answers

    def score(self, generations: list[str], truncateds: list[bool], answer: Any, *, pass_at_k: bool = False) -> tuple[float, tuple[Any, ...], np.ndarray]:
        return score_generations(self, generations, truncateds, answer, pass_at_k=pass_at_k)

    def score_single(self, generation: str, truncated: bool, answer: Any) -> tuple[float, int | None]:
        if truncated:
            return 0.0, None
        model_answer_raw, _ = extract_model_answer(generation, answer_format=self.answer_format)
        try:
            model_answer = int(model_answer_raw) if model_answer_raw is not None else None
        except ValueError:
            model_answer = None
        return (1.0 if model_answer == int(answer) else 0.0), model_answer


class MathTask:
    def __init__(
        self,
        *,
        batch_size: int,
        dataset_name: str,
        seed: int = 0,
        dataset_size: int | None = None,
        answer_format: str = "none",
        apply_chat_template: bool = False,
        tokenizer: Any | None = None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.dataset_name = str(dataset_name).lower()
        self.answer_format = str(answer_format)
        self.apply_chat_template = bool(apply_chat_template)
        self.tokenizer = tokenizer
        self.idx = 0
        self.dataset, self.is_train, self.split_names = _load_math_dataset(
            self.dataset_name,
            seed=int(seed),
            dataset_size=dataset_size,
        )

    def get_batch(self) -> tuple[list[str], list[str]]:
        if not self.is_train:
            raise ValueError(f"get_batch requires a train dataset, got {self.dataset_name!r}.")
        indices = np.arange(self.idx, self.idx + self.batch_size) % len(self.dataset)
        self.idx += self.batch_size
        examples = [self.dataset[int(i)] for i in indices]
        return self._format_examples(examples)

    def get_eval_batch(self) -> tuple[list[str], list[str]]:
        if self.is_train:
            raise ValueError(f"get_eval_batch requires an eval dataset, got {self.dataset_name!r}.")
        indices = np.arange(self.batch_size)
        examples = []
        for split in self.split_names:
            split_dataset = self.dataset[split]
            split_length = len(split_dataset)
            examples.extend(split_dataset[int(i % split_length)] for i in indices)
        return self._format_examples(examples)

    def state_dict(self) -> dict[str, int]:
        return {"idx": int(self.idx)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.idx = int(state.get("idx", 0))

    def score(self, generations: list[str], truncateds: list[bool], answer: Any, *, pass_at_k: bool = False) -> tuple[float, tuple[Any, ...], np.ndarray]:
        return score_generations(self, generations, truncateds, answer, pass_at_k=pass_at_k)

    def score_single(self, generation: str, truncated: bool, answer: Any) -> tuple[float, str | None]:
        if truncated and not _has_explicit_math_answer(generation, answer_format=self.answer_format):
            return 0.0, None
        is_correct, model_answer = check_math_correct(generation, answer, answer_format=self.answer_format)
        return (1.0 if is_correct else 0.0), model_answer

    def _format_examples(self, examples: list[Any]) -> tuple[list[str], list[str]]:
        prompts = [self._format_prompt(example) for example in examples]
        answers = [example["answer"] for example in examples]
        return prompts, answers

    def _format_prompt(self, example: Any) -> str:
        instruction = (
            "Please reason step-by-step concisely."
            if self.answer_format == "answer_tags"
            else "Please reason step-by-step concisely, and put your final answer within \\boxed{ }."
        )
        problem = f"{example['problem']}\n{instruction}"
        if self.apply_chat_template:
            if self.tokenizer is None:
                raise ValueError("apply_chat_template=True requires a tokenizer.")
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": problem}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"User: {problem}\nAssistant: <think>"


class CountdownTask:
    def __init__(
        self,
        *,
        batch_size: int,
        seed: int = 0,
        dataset_size: int | None = None,
        data_path: str | None = None,
        end_token: str | None = None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.end_token = end_token
        self.idx = 0
        self.dataset = _load_countdown_dataset(seed=int(seed), dataset_size=dataset_size, data_path=data_path)

    def get_batch(self) -> tuple[list[str], list[tuple[list[int], int]]]:
        indices = np.arange(self.idx, self.idx + self.batch_size) % len(self.dataset)
        self.idx += self.batch_size
        examples = [self.dataset[int(i)] for i in indices]
        prompts = [example["context"] for example in examples]
        answers = [(list(example["numbers"]), int(example["target"])) for example in examples]
        return prompts, answers

    def state_dict(self) -> dict[str, int]:
        return {"idx": int(self.idx)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.idx = int(state.get("idx", 0))

    def score(self, generations: list[str], truncateds: list[bool], answer: Any, *, pass_at_k: bool = False) -> tuple[float, tuple[Any, ...], np.ndarray]:
        return score_generations(self, generations, truncateds, answer, pass_at_k=pass_at_k)

    def score_single(self, generation: str, truncated: bool, answer: tuple[list[int], int]) -> tuple[float, str | None]:
        if truncated:
            return 0.0, None
        numbers, target = answer
        format_reward = countdown_format_reward("<think>" + generation, self.end_token)
        answer_reward, model_answer = countdown_answer_reward(generation, numbers=numbers, target=target)
        return (format_reward * 0.1 + answer_reward), model_answer


def build_task(
    spec: LLMEnvSpec,
    *,
    batch_size: int,
    seed: int,
    max_tokens: int,
    dataset_size: int | None = None,
    tokenizer: Any | None = None,
    apply_chat_template: bool = False,
) -> LLMTask:
    if spec.task_kind == "zeros":
        return ZerosTask(batch_size=batch_size, max_tokens=max_tokens)
    if spec.task_kind == "random":
        return RandomTask(batch_size=batch_size, max_random_number=4, seed=seed, answer_format=spec.answer_format)
    if spec.task_kind == "math":
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
    if spec.task_kind == "countdown":
        return CountdownTask(batch_size=batch_size, seed=seed, dataset_size=dataset_size)
    raise ValueError(f"Unsupported LLM task kind: {spec.task_kind}")


def _load_math_dataset(dataset_name: str, *, seed: int, dataset_size: int | None) -> tuple[Any, bool, list[str]]:
    dataset_names = {
        "gsm8k": ("axon-rl/GSM-8k", "train", True),
        "asdiv2k": ("axon-rl/ASDIV-2k", "train", True),
        "math12k": ("axon-rl/MATH-12k", "train", True),
        "orz57k": ("axon-rl/ORZ-57k", "train", True),
        "deepscaler40k": ("axon-rl/DeepScaleR-40K", "train", True),
        "math-eval": ("axon-rl/math-eval", ["math", "amc", "olympiad_bench", "minerva", "aime24"], False),
    }
    if dataset_name not in dataset_names:
        raise ValueError(f"Unknown math dataset {dataset_name!r}. Supported: {sorted(dataset_names)}")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("LLM math tasks require the `datasets` package.") from exc

    hf_name, splits, is_train = dataset_names[dataset_name]
    if is_train:
        dataset = load_dataset(hf_name, split=splits).shuffle(seed=seed)
        if dataset_size is not None:
            dataset = dataset.select(range(min(int(dataset_size), len(dataset))))
        return dataset, True, []

    split_names = list(splits)
    raw = dict(load_dataset(hf_name))
    raw["gsm8k"] = load_dataset("axon-rl/GSM-8k", split="train").shuffle(seed=seed).select(range(500))
    raw["asdiv"] = load_dataset("axon-rl/ASDIV-2k", split="train").shuffle(seed=seed).select(range(500))
    raw["aime25"] = load_dataset("math-ai/aime25", split="test").shuffle(seed=seed)
    split_names.extend(["gsm8k", "asdiv", "aime25"])
    if dataset_size is not None:
        limit = int(dataset_size)
        raw = {name: data.select(range(min(limit, len(data)))) for name, data in raw.items()}
    return {name: data.shuffle(seed=seed) for name, data in raw.items()}, False, split_names


def _load_countdown_dataset(*, seed: int, dataset_size: int | None, data_path: str | None) -> Any:
    paths = [Path(data_path)] if data_path else []
    paths.extend([Path("countdown.json"), Path("data/countdown.json")])
    for path in paths:
        if path.exists():
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise RuntimeError("Countdown task JSON loading requires the `datasets` package.") from exc
            dataset = load_dataset("json", data_files=str(path), split="train").shuffle(seed=seed)
            if dataset_size is not None:
                dataset = dataset.select(range(min(int(dataset_size), len(dataset))))
            return dataset
    return _synthetic_countdown_dataset(seed=seed, dataset_size=dataset_size)


def _synthetic_countdown_dataset(*, seed: int, dataset_size: int | None) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    size = int(dataset_size or 1024)
    examples: list[dict[str, Any]] = []
    for _ in range(size):
        numbers = rng.integers(2, 10, size=3).astype(int).tolist()
        target = int(numbers[0] * (numbers[1] + numbers[2]))
        context = (
            "Use each number exactly once to make the target. "
            "Put reasoning in <think> tags and the final expression in <answer> tags.\n"
            f"Numbers: {numbers}\nTarget: {target}\nAssistant: <think>"
        )
        examples.append({"context": context, "numbers": numbers, "target": target})
    return examples


def check_math_correct(generation: str, answer: Any, *, answer_format: str = "none") -> tuple[bool, str | None]:
    correct_answers = [str(x) for x in answer] if isinstance(answer, list) else [str(answer)]
    model_answer = _extract_math_answer(generation, answer_format=answer_format)
    if model_answer is None:
        return False, None
    for correct_answer in correct_answers:
        if _grade_math_answer(model_answer, correct_answer):
            return True, model_answer
    return False, model_answer


def countdown_format_reward(response: str, end_token: str | None = None) -> float:
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]
    think_match = re.search(r"<think>.*?</think>", response, re.DOTALL)
    answer_match = re.search(r"<answer>.*?</answer>", response, re.DOTALL)
    full_format_match = re.match(r"^<think>.*?</think>\n<answer>.*?</answer>$", response, re.DOTALL)
    if full_format_match:
        return 1.0
    reward = 0.0
    if think_match:
        reward += 0.1
    if answer_match:
        reward += 0.5
    return reward


def countdown_answer_reward(response: str, *, numbers: list[int], target: int) -> tuple[float, str | None]:
    matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not matches:
        return 0.0, None
    answer_content = matches[-1].strip()
    if not answer_content or not re.match(r"^[0-9+\-*/() ]+$", answer_content):
        return 0.0, answer_content
    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(int(n) for n in numbers):
        return 0.0, answer_content
    try:
        value = _safe_arithmetic_eval(answer_content)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return 0.0, answer_content
    return (1.0 if abs(float(value) - float(target)) < 1e-5 else 0.0), answer_content


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
    i = start + len(marker)
    depth = 1
    out = []
    while i < len(text):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                answer = "".join(out).strip()
                return answer or None
        out.append(char)
        i += 1
    return None


def _grade_math_answer_with_math_verify(model_answer: str, correct_answer: str) -> bool:
    try:
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
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


def _first_nonempty_match(match: str | tuple[str, ...]) -> str:
    if isinstance(match, tuple):
        for item in match:
            if item:
                return item.strip()
        return ""
    return str(match).strip()


_AST_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_AST_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_arithmetic_eval(expression: str) -> float:
    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _AST_BIN_OPS:
            return float(_AST_BIN_OPS[type(node.op)](eval_node(node.left), eval_node(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _AST_UNARY_OPS:
            return float(_AST_UNARY_OPS[type(node.op)](eval_node(node.operand)))
        raise ValueError(f"Unsupported arithmetic expression: {expression}")

    return eval_node(ast.parse(expression, mode="eval"))
