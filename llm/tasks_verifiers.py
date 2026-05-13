from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np

from llm.tasks_base import score_generations
from llm.tasks_verifiers_utils import (
    answer_payload,
    ensure_supported_env,
    format_prompt,
    get_environment,
    make_state,
    parse_model_answer,
    row_at,
    run_async,
    text_to_assistant_message,
)


_ENV_CACHE: dict[tuple[str, tuple[tuple[str, Any], ...]], Any] = {}


class _VLLMRLMClient:
    """Bridges our vLLM engine to the OpenAI-style client expected by verifiers.v1.RLM."""

    def __init__(
        self,
        llm: Any,
        lora_spec: tuple[str, int, str] | None,
        sampling_params_kwargs: dict[str, Any],
        *,
        tokenizer: Any | None = None,
        apply_chat_template: bool = False,
        env: Any = None,
    ):
        self.llm = llm
        self.lora_spec = lora_spec
        self.sampling_params_kwargs = sampling_params_kwargs
        self.tokenizer = tokenizer
        self.apply_chat_template = apply_chat_template
        self.env = env
        self.chat = self  # For client.chat.completions.create

    @property
    def completions(self):
        return self

    async def create(self, messages: list[dict[str, Any]], **kwargs) -> Any:
        import uuid
        from types import SimpleNamespace

        from llm.vllm_actor_config import lora_requests, sampling_params

        # Convert chat messages back to a prompt string
        if self.apply_chat_template and self.tokenizer is not None:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = ""
            for m in messages:
                role = str(m.get("role", "user")).title()
                content = m.get("content", "")
                prompt += f"{role}: {content}\n"
            if not prompt.endswith("Assistant:\n"):
                prompt += "Assistant:"

        lora_req = lora_requests([self.lora_spec])[0] if self.lora_spec else None
        sampling_params_obj = sampling_params(self.sampling_params_kwargs)

        # Detect if we are using the AsyncEngine or the synchronous LLM engine
        is_async_engine = hasattr(self.llm, "add_request")

        final_output = None
        if is_async_engine:
            request_id = str(uuid.uuid4())
            async for request_output in self.llm.generate(prompt, sampling_params_obj, request_id, lora_request=lora_req):
                final_output = request_output
        else:
            # Synchronous vllm.LLM. Note: LLM.generate expects a list of prompts.
            outputs = self.llm.generate([prompt], sampling_params_obj, lora_request=lora_req, use_tqdm=False)
            if outputs:
                final_output = outputs[0]

        if final_output is None or not final_output.outputs:
            raise RuntimeError("vLLM generation returned no output.")

        # If multiple samples were requested (n > 1), vLLM returns multiple outputs.
        # Currently, the verifiers RLM expectation is a single trajectory, so we
        # provide the first completion.
        best_output = final_output.outputs[0]
        text = best_output.text
        # Use helper to parse Markdown code blocks into ToolCalls if present
        assistant_message = text_to_assistant_message(text, self.env)

        # Return a mock OpenAI response object
        choice = SimpleNamespace(message=assistant_message)
        return SimpleNamespace(choices=[choice], usage=SimpleNamespace(total_tokens=0))


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


class VerifiersTask:
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
        from verifiers.v1 import RLM

        env = self._get_env()
        # Initialize the verifiers Reinforcement Learning Model (RLM)
        client = _VLLMRLMClient(
            llm,
            lora_request_specs[0] if lora_request_specs else None,
            sampling_params_kwargs,
            tokenizer=self.tokenizer,
            apply_chat_template=self.apply_chat_template,
            env=env,
        )
        rlm = RLM(env=env, client=client)

        async def _rollout_and_log(idx: int, payload: Any, prompt: str):
            state = await rlm.rollout(payload)
            reward = float(state.get("reward", 0.0) or 0.0)

            # Format a log string for the trajectory
            traj = state.get("trajectory", [])
            log_parts = [f"PROMPT: {prompt}", f"REWARD: {reward}", "TRAJECTORY:"]
            for step in traj:
                role = getattr(step, "role", "unknown")
                content = getattr(step, "content", "")
                log_parts.append(f"{role.upper()}: {content}")
            return reward, "\n".join(log_parts)

        # Parallelize rollouts using asyncio.gather
        tasks = [_rollout_and_log(i, answers[i], prompts[i]) for i in range(len(prompts))]
        results = await asyncio.gather(*tasks)

        fitnesses = [r[0] for r in results]
        logs = [r[1] for r in results]

        return fitnesses, {}, logs

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
