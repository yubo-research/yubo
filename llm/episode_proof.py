from __future__ import annotations

import base64
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from llm.episodes import Case, RuntimeConfig, Signal, Turn, monotonic_s
from llm.thm_sandbox import build_sandbox_create_request, is_prime_unauthorized_error
from llm.thm_verifiers_env import _has_formal_placeholder
from llm.verifiers_client import VLLMChatAdapter

logger = logging.getLogger(__name__)


class ProofEpisode:
    name = "proof"

    def __init__(
        self,
        env: Any,
        sandbox_client: Any,
        *,
        tokenizer: Any | None = None,
        console: Any | None = None,
        client_factory: Callable[..., Any] = VLLMChatAdapter,
    ) -> None:
        self.env = env
        self.sandbox_client = sandbox_client
        self.tokenizer = tokenizer
        self.console = console
        self.client_factory = client_factory

    async def run(
        self,
        case: Case,
        policy: Any,
        sampling: dict[str, Any],
        runtime: RuntimeConfig,
    ) -> Signal:
        del runtime
        sandbox = await self._create_sandbox(case)
        sandbox_id = sandbox.id
        try:
            return await self._run_in_sandbox(case, policy, sampling, sandbox_id)
        finally:
            try:
                await self.sandbox_client.delete(sandbox_id)
            except Exception:
                logger.warning("Failed to delete theorem sandbox %s", sandbox_id, exc_info=True)

    async def _run_in_sandbox(
        self,
        case: Case,
        policy: Any,
        sampling: dict[str, Any],
        sandbox_id: str,
    ) -> Signal:
        t0 = monotonic_s()
        turns = _initial_turns(self.env, case)
        setup_error = await self._setup_proof(sandbox_id, case, turns, t0)
        if setup_error is not None:
            return setup_error

        client = self.client_factory(
            policy,
            case.metadata.get("lora_spec"),
            sampling,
            tokenizer=self.tokenizer,
            env=self.env,
        )
        messages = _initial_messages(self.env, case)
        state = _ProofState()

        for turn_idx in range(max(1, int(getattr(self.env, "max_turns", 1)))):
            await self._run_turn(client, messages, turns, state, turn_idx, case, sandbox_id)
            if state.done:
                break

        return _final_signal(case, turns, state, monotonic_s() - t0)

    async def _setup_proof(self, sandbox_id: str, case: Case, turns: list[Turn], t0: float) -> Signal | None:
        try:
            await self.sandbox_client.wait_for_creation(sandbox_id)
            await self.env.setup_initial_proof(sandbox_id, case.target, self.sandbox_client)
        except Exception as exc:
            return _setup_failure_signal(case, turns, exc, monotonic_s() - t0)
        return None

    async def _run_turn(
        self,
        client: Any,
        messages: list[dict[str, Any]],
        turns: list[Turn],
        state: "_ProofState",
        turn_idx: int,
        case: Case,
        sandbox_id: str,
    ) -> None:
        msg, content, latency_s = await _create_candidate(client, messages)
        turns.append(Turn(kind="model", name="candidate", text=content, latency_s=latency_s))
        await _broadcast(self.console, turn_idx, {"role": "assistant", "content": content})
        messages.append({"role": "assistant", "content": content})

        tool_calls = _tool_calls(self.env, msg, content, case)
        if not tool_calls:
            state.status = "bad_candidate"
            state.done = True
            return

        for tool_call in tool_calls:
            await self._execute_tool(tool_call, messages, turns, state, content, turn_idx, sandbox_id)
        await self._score_state(case, sandbox_id, state)

    async def _execute_tool(
        self,
        tool_call: Any,
        messages: list[dict[str, Any]],
        turns: list[Turn],
        state: "_ProofState",
        content: str,
        turn_idx: int,
        sandbox_id: str,
    ) -> None:
        state.compile_calls += int(tool_call.name in {self.env.lang_cfg.name, self.env.lang_cfg.extension})
        tool_t0 = monotonic_s()
        output = await self.env.execute_tool(tool_call, sandbox_id, self.sandbox_client)
        state.tool_error = state.tool_error or output.startswith("Error executing tool:")
        state.placeholder = state.placeholder or _has_formal_placeholder(content) or _has_formal_placeholder(output)
        turns.append(
            Turn(
                kind="tool",
                name=str(tool_call.name),
                text=output,
                latency_s=monotonic_s() - tool_t0,
                data={"tool_error": state.tool_error, "placeholder": state.placeholder},
            )
        )
        messages.append({"role": "tool", "content": output, "tool_call_id": tool_call.id})
        await _broadcast(self.console, turn_idx, {"role": "tool", "name": tool_call.name, "output": output})

    async def _score_state(self, case: Case, sandbox_id: str, state: "_ProofState") -> None:
        state.reward = await self.env.rubric.score_state(
            {
                "sandbox_client": self.sandbox_client,
                "sandbox_id": sandbox_id,
                "expected_statement": case.target["statement"],
            }
        )
        if state.reward > 0.0:
            state.status = "ok"
            state.done = True
        elif state.tool_error:
            state.status = "tool_error"
            state.done = True
        elif state.placeholder:
            state.status = "placeholder"
            state.done = True

    async def _create_sandbox(self, case: Case) -> Any:
        req = build_sandbox_create_request(
            docker_image=self.env.docker_image,
            name=f"thm-{self.env.lang_cfg.name}-{_short_case_id(case.id)}-{uuid.uuid4().hex[:10]}",
        )
        try:
            return await self.sandbox_client.create(req)
        except Exception as exc:
            if not is_prime_unauthorized_error(exc):
                raise
            raise RuntimeError("Prime sandbox API rejected the configured credentials. Check PRIME_API_KEY or use THM_SANDBOX_BACKEND=local_docker.") from exc


def _short_case_id(case_id: str) -> str:
    encoded = base64.b32encode(case_id.encode("utf-8")).decode("ascii").lower()
    cleaned = "".join(ch for ch in encoded if ch.isalnum())
    return cleaned[:20] or "case"


@dataclass
class _ProofState:
    compile_calls: int = 0
    tool_error: bool = False
    placeholder: bool = False
    status: str = "wrong"
    reward: float = 0.0
    done: bool = False


def _initial_turns(env: Any, case: Case) -> list[Turn]:
    return [
        Turn(kind="system", name="prompt", text=env.system_prompt()),
        Turn(kind="user", text=env.user_prompt(str(case.prompt), case.target)),
    ]


def _initial_messages(env: Any, case: Case) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": env.system_prompt()},
        {"role": "user", "content": env.user_prompt(str(case.prompt), case.target)},
    ]


def _setup_failure_signal(case: Case, turns: list[Turn], exc: Exception, latency_s: float) -> Signal:
    status = "tool_error"
    turns.append(Turn(kind="tool", name="setup", text=str(exc), data={"tool_error": True}))
    turns.append(Turn(kind="system", name="final", text=status, data={"status": status}))
    return Signal(
        reward=0.0,
        status=status,
        turns=tuple(turns),
        metrics={
            "case_id": case.id,
            "latency_s": latency_s,
            "turns": len(turns),
            "compile_calls": 0,
            "tool_error": True,
            "placeholder": False,
            "verified": False,
        },
        error=str(exc),
    )


async def _create_candidate(client: Any, messages: list[dict[str, Any]]) -> tuple[Any, str, float]:
    model_t0 = monotonic_s()
    res = await client.create(messages)
    msg = res.choices[0].message
    content = str(getattr(msg, "content", "") or "")
    return msg, content, monotonic_s() - model_t0


def _tool_calls(env: Any, msg: Any, content: str, case: Case) -> list[Any]:
    tool_calls = list(getattr(msg, "tool_calls", None) or [])
    if not tool_calls and env.lang_cfg.name == "lean4":
        synthetic_tool = env._lean_tool_call_from_text(content, case.target)
        if synthetic_tool is not None:
            tool_calls = [synthetic_tool]
    return tool_calls


async def _broadcast(console: Any | None, turn_idx: int, step: dict[str, Any]) -> None:
    if console:
        await console.broadcast_step(turn_idx, step)


def _final_signal(case: Case, turns: list[Turn], state: _ProofState, latency_s: float) -> Signal:
    if state.status == "wrong" and state.placeholder:
        state.status = "placeholder"
    turns.append(Turn(kind="system", name="final", text=state.status, data={"status": state.status}))
    return Signal(
        reward=float(state.reward),
        status=state.status,
        turns=tuple(turns),
        metrics={
            "case_id": case.id,
            "latency_s": latency_s,
            "turns": len(turns),
            "compile_calls": state.compile_calls,
            "tool_error": state.tool_error,
            "placeholder": state.placeholder,
            "verified": state.reward > 0.0,
        },
    )


__all__ = ["ProofEpisode"]
