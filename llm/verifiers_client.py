from __future__ import annotations

import json
import time
import uuid
from types import SimpleNamespace
from typing import Any

from llm.model_client import AdapterRef, SampleCall, Sampler
from llm.tasks_verifiers_utils import text_to_assistant_message
from llm.vllm_model_client import VLLMSampler


class ChatAdapter:
    """OpenAI-shaped adapter backed by the repo sampler protocol."""

    def __init__(
        self,
        sampler: Sampler,
        sampling_params_kwargs: dict[str, Any] | None = None,
        *,
        tokenizer: Any | None = None,
        apply_chat_template: bool = False,
        env: Any = None,
    ) -> None:
        self.sampler = sampler
        self.sampling_params_kwargs = dict(sampling_params_kwargs or {})
        self.tokenizer = tokenizer
        self.apply_chat_template = bool(apply_chat_template)
        self.env = env
        self.chat = self
        self._renderer: Any | None = None

    @property
    def completions(self):
        return self

    async def create(self, messages: list[dict[str, Any]], **kwargs) -> Any:
        prompt = self.format_messages_for_generation(messages)
        prompt = self._truncate_prompt_to_context(prompt)
        sampling = dict(self.sampling_params_kwargs)
        sampling.update({key: value for key, value in kwargs.items() if value is not None})
        self._add_renderer_stop_tokens(sampling)
        response = await self._sample(SampleCall(prompt=prompt, sampling=sampling))
        if not response.samples:
            raise RuntimeError("Sampler returned no completions.")

        best_sample = response.samples[0]
        assistant_message = self._assistant_message_from_sample(best_sample)
        choice = SimpleNamespace(message=assistant_message, finish_reason=best_sample.finish_reason)
        usage = SimpleNamespace(
            prompt_tokens=response.usage.prompt_tokens,
            reasoning_tokens=response.usage.reasoning_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
        return SimpleNamespace(
            id=response.request_id,
            created=response.created,
            model=response.model,
            choices=[choice],
            usage=usage,
        )

    def format_messages_for_generation(self, messages: list[dict[str, Any]]) -> str:
        env_formatter = getattr(self.env, "format_messages_for_generation", None)
        env_prompt = env_formatter(messages) if callable(env_formatter) else None
        if env_prompt is not None:
            return str(env_prompt)
        if self.apply_chat_template and self.tokenizer is not None:
            normalized = [message_to_dict(message) for message in messages]
            token_ids = self._get_renderer().render_ids(normalized, add_generation_prompt=True)
            return decode_token_ids(self.tokenizer, token_ids)
        prompt = ""
        for message in messages:
            role = str(message.get("role", "user")).title()
            content = message.get("content", "")
            prompt += f"{role}: {content}\n"
        if not prompt.endswith("Assistant:\n"):
            prompt += "Assistant:"
        return prompt

    def _truncate_prompt_to_context(self, prompt: str) -> str:
        if self.tokenizer is None:
            return prompt
        llm = getattr(self.sampler, "llm", None)
        engine = getattr(llm, "llm_engine", None)
        model_config = getattr(engine, "model_config", None)
        max_model_len = int(getattr(model_config, "max_model_len", 8192))
        max_new_tokens = int(self.sampling_params_kwargs.get("max_tokens", 0) or 0)
        max_input_tokens = max(1, max_model_len - max_new_tokens - 8)
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) <= max_input_tokens:
            return prompt
        token_ids = token_ids[-max_input_tokens:]
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def _add_renderer_stop_tokens(self, sampling: dict[str, Any]) -> None:
        if not self._uses_renderer() or "stop_token_ids" in sampling:
            return
        stop_token_ids = self._get_renderer().get_stop_token_ids()
        if stop_token_ids:
            sampling["stop_token_ids"] = [int(token_id) for token_id in stop_token_ids]

    def _assistant_message_from_sample(self, sample: Any) -> Any:
        if not self._uses_renderer() or not getattr(sample, "token_ids", None):
            return text_to_assistant_message(sample.text, self.env)
        try:
            parsed = self._get_renderer().parse_response(list(sample.token_ids))
        except Exception:
            return text_to_assistant_message(sample.text, self.env)

        if getattr(parsed, "tool_calls", None):
            return assistant_message_from_parsed_response(parsed, sample.text)
        assistant_message = text_to_assistant_message(str(getattr(parsed, "content", sample.text)), self.env)
        reasoning_content = getattr(parsed, "reasoning_content", None)
        if reasoning_content is not None:
            try:
                assistant_message.reasoning_content = str(reasoning_content)
            except Exception:
                pass
        return assistant_message

    def _get_renderer(self) -> Any:
        if self._renderer is None:
            self._renderer = create_message_renderer(self.tokenizer)
        return self._renderer

    def _uses_renderer(self) -> bool:
        return self.apply_chat_template and self.tokenizer is not None

    async def _sample(self, call: SampleCall) -> Any:
        sample_fn = getattr(self.sampler, "sample", None)
        if callable(sample_fn):
            return await sample_fn(call)
        generate_fn = getattr(self.sampler, "generate", None)
        if callable(generate_fn):
            return await generate_fn(call)
        raise TypeError("ChatAdapter requires a sampler with sample(call) or generate(call).")


class VLLMChatAdapter(ChatAdapter):
    """Compatibility shim for old task code/tests that pass a raw vLLM engine."""

    def __init__(
        self,
        llm: Any,
        lora_spec: tuple[str, int, str] | None,
        sampling_params_kwargs: dict[str, Any],
        *,
        tokenizer: Any | None = None,
        apply_chat_template: bool = False,
        env: Any = None,
    ) -> None:
        super().__init__(
            VLLMSampler(
                llm,
                default_sampling=sampling_params_kwargs,
                default_adapter=AdapterRef.from_tuple(lora_spec),
            ),
            sampling_params_kwargs={},
            tokenizer=tokenizer,
            apply_chat_template=apply_chat_template,
            env=env,
        )
        self.llm = llm
        self.lora_spec = lora_spec
        self.sampling_params_kwargs = dict(sampling_params_kwargs)


def has_rollout(env: Any) -> bool:
    return callable(getattr(env, "rollout", None))


def as_verifiers_client(adapter: ChatAdapter) -> Any:
    from verifiers.clients.client import Client
    from verifiers.types import Response, ResponseMessage, Tool, Usage

    class RepoVerifiersClient(Client):
        def __init__(self, chat_adapter: ChatAdapter) -> None:
            super().__init__(chat_adapter)

        def setup_client(self, config):
            return config

        async def to_native_tool(self, tool: Tool) -> dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }

        async def to_native_prompt(self, messages) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            return [message_to_dict(message) for message in messages], {}

        async def get_native_response(
            self,
            prompt: list[dict[str, Any]],
            model: str,
            sampling_args: dict[str, Any],
            tools: list[dict[str, Any]] | None = None,
            **kwargs,
        ) -> Any:
            del model, tools, kwargs
            return await self.client.create(prompt, **dict(sampling_args or {}))

        async def raise_from_native_response(self, response: Any) -> None:
            if response is None or not getattr(response, "choices", None):
                raise RuntimeError("Sampler returned no choices.")

        async def from_native_response(self, response: Any) -> Response:
            choice = response.choices[0]
            raw_message = choice.message
            raw_data = message_to_dict(raw_message)
            finish_reason = getattr(choice, "finish_reason", None) or raw_data.get("finish_reason") or "stop"
            if finish_reason not in {"stop", "length", "tool_calls"}:
                finish_reason = "stop"
            message = ResponseMessage(
                content=raw_data.get("content"),
                reasoning_content=raw_data.get("reasoning_content"),
                thinking_blocks=raw_data.get("thinking_blocks"),
                tool_calls=raw_data.get("tool_calls"),
                finish_reason=finish_reason,
                is_truncated=bool(raw_data.get("is_truncated") or finish_reason == "length"),
                tokens=None,
            )
            usage_obj = getattr(response, "usage", None)
            usage = Usage(
                prompt_tokens=int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                reasoning_tokens=int(getattr(usage_obj, "reasoning_tokens", 0) or 0),
                completion_tokens=int(getattr(usage_obj, "completion_tokens", 0) or 0),
                total_tokens=int(getattr(usage_obj, "total_tokens", 0) or 0),
            )
            return Response(
                id=str(getattr(response, "id", f"model-{uuid.uuid4()}")),
                created=int(getattr(response, "created", int(time.time()))),
                model=str(getattr(response, "model", "runtime.model")),
                usage=usage,
                message=message,
            )

        async def close(self) -> None:
            return None

    return RepoVerifiersClient(adapter)


def message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    out: dict[str, Any] = {}
    for key in (
        "role",
        "content",
        "reasoning_content",
        "thinking_blocks",
        "tool_calls",
        "finish_reason",
        "is_truncated",
    ):
        if hasattr(message, key):
            value = getattr(message, key)
            if value is not None:
                out[key] = value
    return out


def create_message_renderer(tokenizer: Any) -> Any:
    from renderers import create_renderer

    return create_renderer(tokenizer)


def decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode(token_ids)


def assistant_message_from_parsed_response(parsed: Any, fallback_text: str) -> Any:
    from verifiers.types import AssistantMessage

    content = getattr(parsed, "content", None)
    kwargs: dict[str, Any] = {"content": str(fallback_text if content is None else content)}
    reasoning_content = getattr(parsed, "reasoning_content", None)
    if reasoning_content is not None:
        kwargs["reasoning_content"] = str(reasoning_content)
    tool_calls = parsed_tool_calls(getattr(parsed, "tool_calls", None))
    if tool_calls:
        kwargs["tool_calls"] = tool_calls

    try:
        return AssistantMessage(**kwargs)
    except TypeError:
        kwargs.pop("reasoning_content", None)
        try:
            return AssistantMessage(**kwargs)
        except TypeError:
            if tool_calls:
                return AssistantMessage(kwargs["content"], tool_calls=tool_calls)
            return AssistantMessage(kwargs["content"])


def parsed_tool_calls(raw_tool_calls: Any) -> list[Any]:
    if not raw_tool_calls:
        return []

    from verifiers.types import ToolCall

    tool_calls = []
    for raw_call in raw_tool_calls:
        call = message_to_dict(raw_call)
        function = call.get("function") or {}
        if not isinstance(function, dict):
            function = message_to_dict(function)
        name = call.get("name") or function.get("name") or ""
        arguments = call.get("arguments", function.get("arguments", {}))
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, sort_keys=True)
        tool_id = str(call.get("id") or call.get("tool_call_id") or f"call_{uuid.uuid4().hex}")
        tool_calls.append(ToolCall(id=tool_id, name=str(name), arguments=arguments))
    return tool_calls


__all__ = [
    "ChatAdapter",
    "VLLMChatAdapter",
    "as_verifiers_client",
    "has_rollout",
    "message_to_dict",
]
