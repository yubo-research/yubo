from __future__ import annotations

import asyncio
import importlib.metadata as metadata
import json
import re
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # Optional dependency; imported lazily at runtime.
    from verifiers.types import State


def require_verifiers_runtime() -> None:
    """Fail fast on known verifiers dependency drift before vLLM startup."""
    missing: list[str] = []
    for dist_name in ("verifiers", "openai-agents", "openai"):
        try:
            metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            missing.append(dist_name)
    if missing:
        raise RuntimeError(
            "Prime Intellect verifiers runtime is not installed correctly. "
            f"Missing package(s): {', '.join(missing)}. "
            "Run `admin/setup-hyperscalees.sh` for the yubo-hyperscalees env."
        )

    problems = _dependency_problems("openai-agents", only={"openai"})
    try:
        metadata.version("vllm")
    except metadata.PackageNotFoundError:
        pass
    else:
        problems.extend(_dependency_problems("vllm", only={"openai"}))
    if problems:
        raise RuntimeError(
            "Prime Intellect verifiers runtime dependency mismatch: " + "; ".join(problems) + ". Fix the yubo-hyperscalees env with "
            "`python -m pip install 'openai==2.24.0' 'openai-agents==0.10.5'`."
        )


def get_environment(env_id: str, dataset_size: int | None, env_cache: dict) -> Any:
    env_args = default_env_args(env_id, dataset_size)
    key = (str(env_id), tuple(sorted(env_args.items())))
    if key in env_cache:
        return env_cache[key]

    from verifiers.utils.env_utils import load_environment

    try:
        from unittest.mock import patch

        # Monkeypatch signal.signal to a no-op because verifiers tries to register
        # handlers in its __init__, which fails in Ray actor threads.
        with patch("signal.signal", return_value=None):
            env = load_environment(str(env_id), **env_args)
    except Exception as exc:
        if "micromamba" in str(exc) or "git" in str(exc):
            raise RuntimeError(
                f"Failed to load verifiers environment {env_id!r}. "
                "Run admin/setup-hyperscalees.sh on the remote, or install "
                "`verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git`."
            ) from exc
        raise

    env_cache[key] = env
    return env


def default_env_args(env_id: str, dataset_size: int | None) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if env_id == "gsm8k":
        if dataset_size is not None:
            args["num_train_examples"] = int(dataset_size)
    return args


def ensure_supported_env(env: Any, env_id: str) -> None:
    if not hasattr(env, "rubric") or not hasattr(env.rubric, "score_rollout"):
        raise TypeError(f"verifiers environment {env_id!r} must have a rubric with score_rollout().")


def format_prompt(raw_prompt: str, tokenizer: Any, apply_chat_template: bool) -> str:
    if not apply_chat_template or tokenizer is None:
        if isinstance(raw_prompt, list):
            lines: list[str] = []
            for msg in raw_prompt:
                if not isinstance(msg, dict):
                    lines.append(str(msg))
                    continue
                role = str(msg.get("role", "user")).title()
                content = msg.get("content", "")
                lines.append(f"{role}: {content}")
            if not lines or not lines[-1].startswith("Assistant:"):
                lines.append("Assistant:")
            return "\n".join(lines)
        return str(raw_prompt)

    messages = prompt_messages(raw_prompt)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not _ends_with_assistant(messages),
    )


def prompt_messages(raw_prompt: Any) -> list[dict[str, Any]]:
    if isinstance(raw_prompt, list):
        messages = []
        for msg in raw_prompt:
            if isinstance(msg, dict):
                messages.append(
                    {
                        "role": str(msg.get("role", "user")),
                        "content": str(msg.get("content", "")),
                    }
                )
            else:
                messages.append({"role": "user", "content": str(msg)})
        return messages
    return [{"role": "user", "content": str(raw_prompt)}]


def _ends_with_assistant(messages: list[dict[str, Any]]) -> bool:
    return bool(messages) and str(messages[-1].get("role", "")).lower() == "assistant"


def answer_payload(row: dict[str, Any]) -> dict[str, Any]:
    # verifiers expectation: payload is a list of messages or a dict with 'prompt'
    # For many environments, the 'row' itself is sufficient if it contains 'prompt' and 'answer'.
    payload = dict(row)
    payload["__verifiers_payload__"] = True
    return payload


def make_state(answer: Any, completion: list[dict[str, str]], truncated: bool) -> State:
    # Optional dependency: only needed when actually scoring verifiers tasks.
    from verifiers.types import State

    # verifiers rubrics commonly expect `state["input"]` to contain the dataset
    # payload (including the reference answer), plus `state["completion"]`.
    if isinstance(answer, dict) and bool(answer.get("__verifiers_payload__")):
        payload = dict(answer)
    else:
        payload = raw_answer_payload(answer)

    state_dict = {
        "input": payload,
        "completion": completion,
        "truncated": bool(truncated),
        "reward": 0.0,
        "trajectory": [],
    }
    # Initialize optional metadata fields if they exist in State definition
    for field in ["timing", "usage", "error", "stop_condition"]:
        state_dict[field] = None

    return State(state_dict)


def raw_answer_payload(answer: Any) -> dict[str, Any]:
    return {
        "__verifiers_payload__": True,
        "answer": str(answer),
    }


def parse_model_answer(env: Any, completion: list[dict[str, str]]) -> Any:
    # Best-effort attempt to parse the final answer using the environment's parser
    if not hasattr(env, "parser") or not hasattr(env.parser, "parse_answer"):
        rubric = getattr(env, "rubric", None)
        parser = getattr(rubric, "parser", None)
        if parser is None or not hasattr(parser, "parse_answer"):
            return completion[-1]["content"] if completion else ""
        return parser.parse_answer(completion)
    return env.parser.parse_answer(completion)


def text_to_assistant_message(text: str, env: Any = None) -> Any:
    """Bridges model text (Markdown code blocks) to AssistantMessage + ToolCalls."""
    from verifiers.types import AssistantMessage, ToolCall

    # 1. Check if model already provided structured tool calls (future-proofing)
    # 2. Fallback to parsing Markdown code blocks
    # We look for all ```lang code ``` blocks, allowing for flexible spacing/newlines.
    blocks = re.findall(r"```(\w+)\s*\n?(.*?)\n?```", text, re.DOTALL)
    if not blocks:
        return AssistantMessage(content=text)

    # Determine default tool name from environment
    default_tool_name = "python"
    if env and hasattr(env, "tools") and env.tools:
        default_tool_name = next(iter(env.tools.keys()))

    tool_calls = []
    for lang, code in blocks:
        tool_id = str(uuid.uuid4())
        # Translate common tags to tool names
        t_name = lang.lower()
        if t_name in ("py", "python3"):
            t_name = "python"

        # Special case for formal languages if the tool name doesn't match exactly
        if t_name not in (default_tool_name, "bash"):
            # Check if it's a known formal language or bash
            if t_name not in ("lean", "lean4", "coq", "isabelle", "bash"):
                t_name = default_tool_name

        tool_calls.append(
            ToolCall(
                id=tool_id,
                name=t_name,
                arguments=json.dumps({"code": code.strip()}),
            )
        )

    return AssistantMessage(content=text, tool_calls=tool_calls)


def _dependency_problems(dist_name: str, *, only: set[str]) -> list[str]:
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
    except Exception:
        return []

    problems: list[str] = []
    for raw_req in metadata.requires(dist_name) or []:
        try:
            req = Requirement(raw_req)
        except Exception:
            continue
        if req.name not in only:
            continue
        if req.marker is not None and not req.marker.evaluate():
            continue
        try:
            installed = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            problems.append(f"{dist_name} requires {req}, but {req.name} is not installed")
            continue
        if req.specifier and Version(installed) not in req.specifier:
            problems.append(f"{dist_name} requires {req}, but {req.name}=={installed} is installed")
    return problems


def _run_async(awaitable: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    if loop.is_running():
        # This is a workaround for environments where the loop is already running (like Ray actors)
        # but we need to run a coroutine synchronously.
        try:
            import nest_asyncio

            nest_asyncio.apply(loop)
        except ImportError:
            pass

    return loop.run_until_complete(awaitable)


def run_async(awaitable: Any) -> Any:
    return _run_async(awaitable)


def row_at(dataset: Any, index: int) -> dict[str, Any]:
    row = dataset[int(index)]
    if not isinstance(row, dict):
        try:
            row = dict(row)
        except Exception as exc:
            raise TypeError(f"verifiers dataset rows must be mapping-like, got {type(row)!r}.") from exc
    return row
