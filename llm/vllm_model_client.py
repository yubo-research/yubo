from __future__ import annotations

from typing import Any

from llm.model_client import AdapterRef, Completion, SampleBatch, SampleCall, TokenUsage
from llm.vllm_actor_config import lora_requests, sampling_params


class VLLMSampler:
    """Repo-owned sampler for sync and async vLLM engines."""

    def __init__(
        self,
        llm: Any,
        *,
        default_sampling: dict[str, Any] | None = None,
        default_adapter: AdapterRef | None = None,
    ) -> None:
        self.llm = llm
        self.default_sampling = dict(default_sampling or {})
        self.default_adapter = default_adapter

    async def sample(self, call: SampleCall) -> SampleBatch:
        sampling_kwargs = dict(self.default_sampling)
        sampling_kwargs.update(dict(call.sampling or {}))
        sampling_obj = sampling_params(sampling_kwargs)
        adapter = call.adapter or self.default_adapter
        lora_req = lora_requests([adapter.as_lora_tuple()])[0] if adapter is not None else None

        output = await self._generate_one(call.prompt, sampling_obj, call.request_id, lora_req)
        if output is None or not getattr(output, "outputs", None):
            raise RuntimeError("vLLM generation returned no output.")

        samples = [
            Completion(
                text=str(getattr(sample, "text", "")),
                finish_reason=str(getattr(sample, "finish_reason", "stop") or "stop"),
                token_ids=tuple(int(token) for token in (getattr(sample, "token_ids", None) or ())),
            )
            for sample in output.outputs
        ]
        usage = _usage_from_output(output)
        return SampleBatch(
            request_id=call.request_id,
            model=call.model,
            samples=samples,
            usage=usage,
            raw=output,
        )

    async def generate(self, request: SampleCall) -> SampleBatch:
        return await self.sample(request)

    async def _generate_one(self, prompt: str, sampling_obj: Any, request_id: str, lora_req: Any) -> Any:
        is_async_engine = hasattr(self.llm, "add_request")
        if is_async_engine:
            final_output = None
            async for request_output in self.llm.generate(prompt, sampling_obj, request_id, lora_request=lora_req):
                final_output = request_output
            return final_output
        outputs = self.llm.generate([prompt], sampling_obj, lora_request=lora_req, use_tqdm=False)
        return outputs[0] if outputs else None


def _usage_from_output(output: Any) -> TokenUsage:
    prompt_token_ids = getattr(output, "prompt_token_ids", None) or ()
    completion_tokens = 0
    for sample in getattr(output, "outputs", None) or ():
        completion_tokens += len(getattr(sample, "token_ids", None) or ())
    prompt_tokens = len(prompt_token_ids)
    return TokenUsage(
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
        total_tokens=int(prompt_tokens + completion_tokens),
    )


__all__ = ["VLLMSampler"]
