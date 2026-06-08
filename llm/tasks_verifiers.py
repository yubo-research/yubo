from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from llm.episode_runner import EpisodeRunner
from llm.episode_verifiers import VerifiersEpisode
from llm.episodes import Case, RuntimeConfig, signal_log, summarize_signals
from llm.tasks_base import RolloutTaskMixin, score_generations
from llm.tasks_verifiers_utils import (
    answer_payload,
    ensure_supported_env,
    format_prompt,
    get_environment,
    make_state,
    parse_model_answer,
    row_at,
    run_async,
)
from llm.verifiers_client import VLLMChatAdapter

_ENV_CACHE: dict[tuple[str, tuple[tuple[str, Any], ...]], Any] = {}

_VLLMRLMClient = VLLMChatAdapter


@dataclass(frozen=True)
class VerifiersTaskConfig:
    batch_size: int
    env_id: str
    seed: int = 0
    dataset_size: int | None = None
    tokenizer: Any | None = None
    apply_chat_template: bool = False

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "VerifiersTaskConfig":
        return cls(**kwargs)


class VerifiersTask(RolloutTaskMixin):
    """Adapter from Prime Intellect verifiers environments to the repo LLMTask API.

    The live verifiers environment is intentionally not pickled. UHD sends task
    objects into Ray actors for scoring, and some verifiers rubrics own process
    pools. Each process lazily loads and caches its own environment instead.
    """

    def __init__(self, config: VerifiersTaskConfig | None = None, **kwargs: Any) -> None:
        cfg = config if config is not None else VerifiersTaskConfig.from_kwargs(**kwargs)
        self.batch_size = int(cfg.batch_size)
        self.env_id = str(cfg.env_id)
        self.seed = int(cfg.seed)
        self.dataset_size = None if cfg.dataset_size is None else int(cfg.dataset_size)
        self.tokenizer = cfg.tokenizer
        self.apply_chat_template = bool(cfg.apply_chat_template)
        self.idx = 0
        self._env = None
        self._dataset = None

    def __getstate__(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "env_id": self.env_id,
            "seed": self.seed,
            "dataset_size": self.dataset_size,
            "apply_chat_template": self.apply_chat_template,
            "idx": self.idx,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.batch_size = int(state["batch_size"])
        self.env_id = str(state["env_id"])
        self.seed = int(state["seed"])
        self.dataset_size = None if state["dataset_size"] is None else int(state["dataset_size"])
        self.tokenizer = None
        self.apply_chat_template = bool(state.get("apply_chat_template", False))
        self.idx = int(state.get("idx", 0))
        self._env = None
        self._dataset = None

    def generate_and_score(
        self,
        llm: Any,
        prompts: list[str],
        sampling_params_kwargs: dict[str, Any],
        lora_request_specs: list[tuple[str, int, str]] | None,
        answers: list[Any],
        args: Any,
    ) -> tuple[list[float], dict[str, float], list[str]]:
        return run_async(self.generate_and_score_async(llm, prompts, sampling_params_kwargs, lora_request_specs, answers, args))

    async def generate_and_score_async(
        self,
        llm: Any,
        prompts: list[str],
        sampling_params_kwargs: dict[str, Any],
        lora_request_specs: list[tuple[str, int, str]] | None,
        answers: list[Any],
        args: Any,
    ) -> tuple[list[float], dict[str, float], list[str]]:
        env = self._get_env()
        episode = VerifiersEpisode(
            env,
            tokenizer=self.tokenizer,
            apply_chat_template=self.apply_chat_template,
        )
        cases = [
            Case(
                id=f"{self.env_id}:{idx}",
                prompt=prompts[idx],
                target=answers[idx],
                metadata={"lora_spec": None if lora_request_specs is None else lora_request_specs[idx]},
            )
            for idx in range(len(prompts))
        ]
        runtime = RuntimeConfig(concurrency=max(1, int(getattr(args, "rollout_concurrency", len(cases) or 1))))
        signals = await EpisodeRunner(runtime).run_batch(episode, cases, llm, dict(sampling_params_kwargs))
        fitnesses = [float(signal.reward) for signal in signals]
        logs = [signal_log(case, signal) for case, signal in zip(cases, signals, strict=True)]
        return fitnesses, summarize_signals(signals), logs

    def get_batch(self) -> tuple[list[str], list[Any]]:
        dataset = self._get_dataset()
        length = len(dataset)
        if length == 0:
            raise ValueError(f"verifiers environment {self.env_id!r} returned an empty dataset.")
        indices = np.arange(self.idx, self.idx + self.batch_size) % length
        self.idx += self.batch_size
        rows = [row_at(dataset, int(index)) for index in indices]
        prompts = [
            format_prompt(
                row.get("prompt", ""),
                tokenizer=self.tokenizer,
                apply_chat_template=self.apply_chat_template,
            )
            for row in rows
        ]
        answers = [answer_payload(row) for row in rows]
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
    ) -> tuple[float, tuple[Any, ...], np.ndarray]:
        return score_generations(self, generations, truncateds, answer, pass_at_k=pass_at_k)

    def score_single(self, generation: str, truncated: bool, answer: Any) -> tuple[float, Any]:
        env = get_environment(self.env_id, self.dataset_size, _ENV_CACHE)
        ensure_supported_env(env, self.env_id)
        completion = [{"role": "assistant", "content": str(generation)}]
        state = make_state(answer, completion=completion, truncated=bool(truncated))
        run_async(env.rubric.score_rollout(state))
        reward = float(state.get("reward", 0.0) or 0.0)
        model_answer = parse_model_answer(env, completion)
        return reward, model_answer

    def _get_dataset(self) -> Any:
        if self._dataset is not None:
            return self._dataset
        env = self._get_env()
        n = -1 if self.dataset_size is None else int(self.dataset_size)
        if not hasattr(env, "get_dataset"):
            raise ValueError(f"verifiers environment {self.env_id!r} does not expose get_dataset().")
        self._dataset = env.get_dataset(n=n, seed=self.seed)
        return self._dataset

    def _get_env(self) -> Any:
        if self._env is None:
            self._env = get_environment(self.env_id, self.dataset_size, _ENV_CACHE)
            ensure_supported_env(self._env, self.env_id)
        return self._env


__all__ = ["VerifiersTask", "VerifiersTaskConfig"]
