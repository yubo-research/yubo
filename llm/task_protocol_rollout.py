from __future__ import annotations

from typing import Any, Protocol

from llm.task_protocol_core import LLMTask


class RolloutTask(LLMTask, Protocol):
    def generate_and_score(
        self,
        llm: Any,
        prompts: list[str],
        sampling_params_kwargs: dict[str, Any],
        lora_request_specs: list[tuple[str, int, str]] | None,
        answers: list[Any],
        args: Any,
    ) -> tuple[list[float], dict[str, float], list[str]]: ...


__all__ = ["RolloutTask"]
