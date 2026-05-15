from __future__ import annotations

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

    @property
    def completions(self):
        return self

    async def create(self, messages: list[dict[str, Any]], **kwargs) -> Any:
        prompt = self.format_messages_for_generation(messages)
        prompt = self._truncate_prompt_to_context(prompt)
        sampling = dict(self.sampling_params_kwargs)
        sampling.update({key: value for key, value in kwargs.items() if value is not None})
        response = await self._sample(SampleCall(prompt=prompt, sampling=sampling))
        if not response.samples:
            raise RuntimeError("Sampler returned no completions.")

        best_sample = response.samples[0]
        assistant_message = text_to_assistant_message(best_sample.text, self.env)
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
            return self.tokenizer.apply_chat_template(normalized, tokenize=False, add_generation_prompt=True)
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


__all__ = [
    "ChatAdapter",
    "VLLMChatAdapter",
    "as_verifiers_client",
    "has_rollout",
    "message_to_dict",
]
