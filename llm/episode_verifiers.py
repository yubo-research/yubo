from __future__ import annotations

from typing import Any

from llm.episodes import Case, RuntimeConfig, Signal, Turn, monotonic_s
from llm.verifiers_client import VLLMChatAdapter, as_verifiers_client, has_rollout


class VerifiersEpisode:
    name = "verifiers"

    def __init__(
        self,
        env: Any,
        *,
        tokenizer: Any | None = None,
        apply_chat_template: bool = False,
    ) -> None:
        self.env = env
        self.tokenizer = tokenizer
        self.apply_chat_template = bool(apply_chat_template)

    async def run(
        self,
        case: Case,
        policy: Any,
        sampling: dict[str, Any],
        runtime: RuntimeConfig,
    ) -> Signal:
        del runtime
        t0 = monotonic_s()
        client = VLLMChatAdapter(
            policy,
            case.metadata.get("lora_spec"),
            sampling,
            tokenizer=self.tokenizer,
            apply_chat_template=self.apply_chat_template,
            env=self.env,
        )
        rollout_client = as_verifiers_client(client) if has_rollout(self.env) else None
        if rollout_client is not None:
            state = await self.env.rollout(
                case.target,
                client=rollout_client,
                model="runtime.model",
                sampling_args=dict(sampling),
            )
        else:
            from verifiers.v1 import RLM

            state = await _rollout_with_rlm(RLM, self.env, client, case.target)

        reward = float(state.get("reward", 0.0) or 0.0)
        turns = tuple(_turns_from_state(state))
        latency_s = monotonic_s() - t0
        return Signal(
            reward=reward,
            status="ok" if reward > 0.0 else "wrong",
            turns=turns,
            metrics={
                "case_id": case.id,
                "latency_s": latency_s,
                "turns": len(turns),
                "uses_env_rollout": rollout_client is not None,
            },
        )


def _turns_from_state(state: Any) -> list[Turn]:
    turns = _turns_from_trajectory(_state_get(state, "trajectory", []))
    if not _has_assistant_text(turns):
        turns.extend(_turns_from_completion(_state_get(state, "completion", [])))
    return turns


async def _rollout_with_rlm(rlm_cls: Any, env: Any, client: Any, target: Any) -> Any:
    errors: list[TypeError] = []
    for factory in (
        lambda: rlm_cls(env=env, client=client),
        lambda: rlm_cls(env, client=client),
        lambda: rlm_cls(client=client),
        lambda: rlm_cls(client),
    ):
        try:
            rlm = factory()
        except TypeError as exc:
            errors.append(exc)
            continue
        return await rlm.rollout(target)
    raise errors[-1]


def _turns_from_trajectory(trajectory: Any) -> list[Turn]:
    turns: list[Turn] = []
    for step in trajectory or []:
        turns.extend(_turns_from_trajectory_step(step))
    return turns


def _turns_from_trajectory_step(step: Any) -> list[Turn]:
    completion = _field(step, "completion")
    if completion is not None:
        turns = _turns_from_completion(completion)
        if _has_text(turns):
            return turns
    response = _field(step, "response")
    if response is not None:
        turn = _turn_from_message(response, default_role="assistant")
        if turn.text.strip():
            return [turn]
    return [_turn_from_message(step, default_role="unknown")]


def _turns_from_completion(completion: Any) -> list[Turn]:
    if completion is None:
        return []
    messages = completion if isinstance(completion, list | tuple) else [completion]
    return [_turn_from_message(message, default_role="assistant") for message in messages]


def _turn_from_message(message: Any, *, default_role: str) -> Turn:
    nested = _field(message, "message")
    source = nested if nested is not None else message
    role = _field(source, "role", _field(message, "role", default_role))
    content = _content_text(source)
    name = _field(source, "name", _field(message, "name", None))
    return Turn(
        kind=str(role or default_role),
        text=content,
        name=None if name is None else str(name),
    )


def _content_text(message: Any) -> str:
    for key in ("content", "output", "text"):
        value = _field(message, key)
        if value is not None:
            return _stringify_content(value)
    return ""


def _field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _state_get(state: Any, key: str, default: Any = None) -> Any:
    getter = getattr(state, "get", None)
    if callable(getter):
        return getter(key, default)
    return _field(state, key, default)


def _stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        parts = [_stringify_content(_field(item, "text", _field(item, "content", item))) for item in value]
        text = "\n".join(part for part in parts if part)
        return text if text else str(value)
    return str(value)


def _has_assistant_text(turns: list[Turn]) -> bool:
    return any(turn.kind.lower() == "assistant" and bool(turn.text.strip()) for turn in turns)


def _has_text(turns: list[Turn]) -> bool:
    return any(bool(turn.text.strip()) for turn in turns)


__all__ = ["VerifiersEpisode"]
