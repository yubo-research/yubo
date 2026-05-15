from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from llm.console_observer import UnifiedConsoleManager
from llm.episode_proof import ProofEpisode
from llm.episode_runner import EpisodeRunner
from llm.episodes import Case, RuntimeConfig, signal_log, summarize_signals
from llm.tasks_base import RolloutTaskMixin
from llm.thm_sandbox import (
    build_sandbox_client,
)
from llm.thm_verifiers_env import LANGUAGES, FormalRubric, LanguageConfig, TheoremVerifierEnv

_VLLMRLMClient: Any | None = None


def _vllm_rlm_client_cls() -> Any:
    global _VLLMRLMClient
    if _VLLMRLMClient is None:
        from llm.tasks_verifiers import _VLLMRLMClient as client_cls

        _VLLMRLMClient = client_cls
    return _VLLMRLMClient


def prime_sandbox_auth_configured(
    *,
    environ: Mapping[str, str] | None = None,
    home: str | Path | None = None,
) -> bool:
    env = os.environ if environ is None else environ
    if env.get("PRIME_API_KEY"):
        return True
    home_path = Path.home() if home is None else Path(home)
    return (home_path / ".prime" / "config.json").is_file()


def prime_sandbox_auth_summary() -> str:
    key = os.environ.get("PRIME_API_KEY")
    if key:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
        return f"PRIME_API_KEY set len={len(key)} sha256={digest}"
    config_path = Path.home() / ".prime" / "config.json"
    if config_path.is_file():
        return f"Prime config present at {config_path}"
    return "no PRIME_API_KEY and no ~/.prime/config.json"


class TheoremProvingTask(RolloutTaskMixin):
    """Universal task for formal theorem proving (Lean, Coq, Isabelle)."""

    def __init__(
        self,
        language: str = "lean4",
        dataset_name: str = "cat-searcher/minif2f-lean4",
        dataset_split: str = "validation",
        batch_size: int = 1,
        seed: int = 0,
        tokenizer: Any | None = None,
        console: UnifiedConsoleManager | None = None,
    ):
        self.lang_cfg = LANGUAGES.get(language)
        if not self.lang_cfg:
            raise ValueError(f"Unsupported language: {language}")
        self.env = TheoremVerifierEnv(self.lang_cfg)

        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.batch_size = batch_size
        self.seed = seed
        self.tokenizer = tokenizer
        self.console = console
        self.idx = 0
        self._dataset = None

    def __getstate__(self) -> dict[str, Any]:
        return {
            "lang_name": self.lang_cfg.name,
            "dataset_name": self.dataset_name,
            "dataset_split": self.dataset_split,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "idx": self.idx,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.lang_cfg = LANGUAGES[state["lang_name"]]
        self.env = TheoremVerifierEnv(self.lang_cfg)
        self.dataset_name = state["dataset_name"]
        self.dataset_split = state["dataset_split"]
        self.batch_size = state["batch_size"]
        self.seed = state["seed"]
        self.idx = state["idx"]
        self.tokenizer = None
        self.console = None
        self._dataset = None

    def get_batch(self) -> tuple[list[str], list[Any]]:
        if self._dataset is None:
            from datasets import load_dataset

            self._dataset = load_dataset(self.dataset_name, split=self.dataset_split)

        indices = np.arange(self.idx, self.idx + self.batch_size) % len(self._dataset)
        self.idx += self.batch_size
        rows = [self._dataset[int(i)] for i in indices]

        prompts = []
        answers = []
        for row in rows:
            stmt = row.get("formal_statement") or row.get("statement") or ""
            prompt = f"Prove the following theorem in {self.lang_cfg.name}:\n\n{stmt}"
            prompts.append(prompt)
            answers.append({"statement": stmt, "row": row})

        return prompts, answers

    async def generate_and_score_async(
        self,
        llm: Any,
        prompts: list[str],
        sampling_params_kwargs: dict[str, Any],
        lora_request_specs: list[tuple[str, int, str]] | None,
        answers: list[Any],
        args: Any,
    ) -> tuple[list[float], dict[str, float], list[str]]:
        sandbox_client = await build_sandbox_client()
        env = self._active_env()
        episode = ProofEpisode(
            env,
            sandbox_client,
            tokenizer=self.tokenizer,
            console=self.console,
            client_factory=_vllm_rlm_client_cls(),
        )
        cases = [
            Case(
                id=f"{self.lang_cfg.name}:{idx}",
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
        self.last_logs = logs[:3]

        return fitnesses, summarize_signals(signals), logs

    def generate_and_score(
        self,
        llm: Any,
        prompts: list[str],
        sampling_params_kwargs: dict[str, Any],
        lora_request_specs: list[tuple[str, int, str]] | None,
        answers: list[Any],
        args: Any,
    ) -> tuple[list[float], dict[str, float], list[str]]:
        return _run_async(
            self.generate_and_score_async(
                llm=llm,
                prompts=prompts,
                sampling_params_kwargs=sampling_params_kwargs,
                lora_request_specs=lora_request_specs,
                answers=answers,
                args=args,
            )
        )

    async def _run_single(self, llm, prompt, sampling, lora_spec, answer, sandbox_client):
        env = self._active_env()
        case = Case(
            id=f"{self.lang_cfg.name}:single",
            prompt=prompt,
            target=answer,
            metadata={"lora_spec": lora_spec},
        )
        signal = await ProofEpisode(
            env,
            sandbox_client,
            tokenizer=self.tokenizer,
            console=self.console,
            client_factory=_vllm_rlm_client_cls(),
        ).run(
            case,
            llm,
            dict(sampling),
            RuntimeConfig(),
        )
        if self.console:
            await self.console.broadcast_reward(signal.reward, {"status": signal.status})
        return signal.reward, signal_log(case, signal)

    async def _setup_initial_proof(self, sandbox_id, answer, sandbox_client):
        await self._active_env().setup_initial_proof(sandbox_id, answer, sandbox_client)

    async def _execute_tool(self, tool_call, sandbox_id, sandbox_client, proof_path):
        return await self._active_env().execute_tool(tool_call, sandbox_id, sandbox_client)

    def score(self, generations, truncateds, answer, *, pass_at_k=False) -> tuple[float, tuple[Any, ...], np.ndarray]:
        # This task primarily uses generate_and_score_async
        return 0.0, (), np.asarray([])

    @property
    def task_name(self) -> str:
        return f"thm:{self.lang_cfg.name}"

    def _active_env(self) -> TheoremVerifierEnv:
        if self.env.lang_cfg != self.lang_cfg:
            self.env = TheoremVerifierEnv(self.lang_cfg)
        return self.env


def _run_async(awaitable: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    if loop.is_running():
        try:
            import nest_asyncio

            nest_asyncio.apply(loop)
        except ImportError:
            pass

    return loop.run_until_complete(awaitable)


__all__ = [
    "TheoremProvingTask",
    "LanguageConfig",
    "LANGUAGES",
    "FormalRubric",
    "prime_sandbox_auth_configured",
    "prime_sandbox_auth_summary",
]
