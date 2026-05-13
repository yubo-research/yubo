from __future__ import annotations

import ast
import operator
import re
from pathlib import Path
from typing import Any

import numpy as np

from llm.tasks_base import score_generations


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

    def score(
        self,
        generations: list[str],
        truncateds: list[bool],
        answer: Any,
        *,
        pass_at_k: bool = False,
    ):
        return score_generations(self, generations, truncateds, answer, pass_at_k=pass_at_k)

    def score_single(self, generation: str, truncated: bool, answer: tuple[list[int], int]) -> tuple[float, str | None]:
        if truncated:
            return 0.0, None
        numbers, target = answer
        format_reward = countdown_format_reward("<think>" + generation, self.end_token)
        answer_reward, model_answer = countdown_answer_reward(generation, numbers=numbers, target=target)
        return (format_reward * 0.1 + answer_reward), model_answer


def _load_countdown_dataset(*, seed: int, dataset_size: int | None, data_path: str | None) -> Any:
    path = _first_existing_countdown_path(data_path)
    if path is None:
        return _synthetic_countdown_dataset(seed=seed, dataset_size=dataset_size)
    return _load_countdown_json_dataset(path, seed=seed, dataset_size=dataset_size)


def _synthetic_countdown_dataset(*, seed: int, dataset_size: int | None) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    examples: list[dict[str, Any]] = []
    for _ in range(int(dataset_size or 1024)):
        numbers = rng.integers(2, 10, size=3).astype(int).tolist()
        target = int(numbers[0] * (numbers[1] + numbers[2]))
        examples.append(
            {
                "context": _countdown_context(numbers, target),
                "numbers": numbers,
                "target": target,
            }
        )
    return examples


def countdown_format_reward(response: str, end_token: str | None = None) -> float:
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]
    think_match = re.search(r"<think>.*?</think>", response, re.DOTALL)
    answer_match = re.search(r"<answer>.*?</answer>", response, re.DOTALL)
    full_format_match = re.match(r"^<think>.*?</think>\n<answer>.*?</answer>$", response, re.DOTALL)
    if full_format_match:
        return 1.0
    return (0.1 if think_match else 0.0) + (0.5 if answer_match else 0.0)


def countdown_answer_reward(response: str, *, numbers: list[int], target: int) -> tuple[float, str | None]:
    matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not matches:
        return 0.0, None
    answer_content = matches[-1].strip()
    if not _is_valid_countdown_expression(answer_content, numbers):
        return 0.0, answer_content
    try:
        value = _safe_arithmetic_eval(answer_content)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return 0.0, answer_content
    return (1.0 if abs(float(value) - float(target)) < 1e-5 else 0.0), answer_content


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


def _first_existing_countdown_path(data_path: str | None) -> Path | None:
    paths = [Path(data_path)] if data_path else []
    paths.extend([Path("countdown.json"), Path("data/countdown.json")])
    return next((path for path in paths if path.exists()), None)


def _load_countdown_json_dataset(path: Path, *, seed: int, dataset_size: int | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Countdown task JSON loading requires the `datasets` package.") from exc
    dataset = load_dataset("json", data_files=str(path), split="train").shuffle(seed=seed)
    if dataset_size is not None:
        dataset = dataset.select(range(min(int(dataset_size), len(dataset))))
    return dataset


def _countdown_context(numbers: list[int], target: int) -> str:
    return (
        "Use each number exactly once to make the target. "
        "Put reasoning in <think> tags and the final expression in <answer> tags.\n"
        f"Numbers: {numbers}\nTarget: {target}\nAssistant: <think>"
    )


def _is_valid_countdown_expression(answer_content: str, numbers: list[int]) -> bool:
    if not answer_content or not re.match(r"^[0-9+\-*/() ]+$", answer_content):
        return False
    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    return sorted(used_numbers) == sorted(int(n) for n in numbers)


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
