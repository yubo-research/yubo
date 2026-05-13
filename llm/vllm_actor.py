from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

from llm.lora import materialize_lora_adapters
from llm.vllm_actor_config import (
    VLLMActorConfig,
    external_import,
    get_llm_kwargs,
    lora_requests,
    sampling_params,
    set_vllm_env_defaults,
)
from llm.vllm_scoring import score_request_outputs


class EggrollVLLMActor:
    def __init__(self, config: VLLMActorConfig | None = None, **kwargs) -> None:
        cfg = config if config is not None else VLLMActorConfig.from_kwargs(**kwargs)
        set_vllm_env_defaults()

        self.llm = external_import("v", "llm").LLM(**get_llm_kwargs(cfg))
        self.rank = 0
        self.adapter_root = Path(os.getenv("YUBO_LORA_POPULATION_PATH", "/dev/shm/yubo_llm_lora_population"))
        self.peft_state_dict = None
        self.peft_shapes_dict = None
        self.lora_config_dict = None

    def collective_rpc(self, method: str, args: tuple[Any, ...] = ()) -> Any:
        return self.llm.collective_rpc(method, args=args)

    def shutdown(self) -> bool:
        return bool(_shutdown_owner_llm(self))

    def __ray_shutdown__(self) -> None:
        _shutdown_owner_llm(self)

    def _shutdown_vllm(self) -> bool:
        return _shutdown_owner_llm(self)

    def setup_local_lora_generation(
        self,
        peft_state_dict: dict[str, Any],
        peft_shapes_dict: dict[str, tuple[int, ...]],
        lora_config_dict: dict[str, Any],
        rank: int,
    ) -> bool:
        self.rank = int(rank)
        self.peft_state_dict = peft_state_dict
        self.peft_shapes_dict = peft_shapes_dict
        self.lora_config_dict = lora_config_dict
        return True

    def generate_local_adapters(self, population_indices: list[int], es_step: int, args: Any) -> list[str]:
        if self.peft_state_dict is None or self.peft_shapes_dict is None or self.lora_config_dict is None:
            raise RuntimeError("LoRA generation was not initialized on this actor.")
        return materialize_lora_adapters(
            adapter_root=self.adapter_root,
            rank=self.rank,
            population_indices=population_indices,
            es_step=int(es_step),
            args=args,
            peft_state_dict=self.peft_state_dict,
            peft_shapes_dict=self.peft_shapes_dict,
            lora_config_dict=self.lora_config_dict,
        )

    def generate_and_score(
        self,
        prompts: list[str],
        sampling_params_kwargs: dict[str, Any],
        lora_request_specs: list[tuple[str, int, str]] | None,
        task_obj: Any,
        answers: list[Any],
        args: Any,
    ) -> tuple[list[float], dict[str, float], list[str]]:
        if hasattr(task_obj, "generate_and_score"):
            return task_obj.generate_and_score(
                llm=self.llm,
                prompts=prompts,
                sampling_params_kwargs=sampling_params_kwargs,
                lora_request_specs=lora_request_specs,
                answers=answers,
                args=args,
            )

        request_outputs = self.llm.generate(
            prompts,
            sampling_params(sampling_params_kwargs),
            lora_request=lora_requests(lora_request_specs),
            use_tqdm=True,
        )
        return score_request_outputs(
            request_outputs,
            prompts=prompts,
            task_obj=task_obj,
            answers=answers,
            pass_at_k=bool(args.pass_at_k),
        )


class TextVLLMActor(EggrollVLLMActor):
    pass


class AsyncTextVLLMActor(EggrollVLLMActor):
    def __init__(self, config: VLLMActorConfig | None = None, **kwargs) -> None:
        cfg = config if config is not None else VLLMActorConfig.from_kwargs(**kwargs)
        set_vllm_env_defaults()

        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        # Async engine does not support Ray distributed backend natively without specific args,
        # but the collective_rpc extension bridges this gap.
        engine_args = AsyncEngineArgs(**get_llm_kwargs(cfg))
        self.llm = AsyncLLMEngine.from_engine_args(engine_args)

        self.rank = 0
        self.adapter_root = Path(os.getenv("YUBO_LORA_POPULATION_PATH", "/dev/shm/yubo_llm_lora_population"))
        self.peft_state_dict = None
        self.peft_shapes_dict = None
        self.lora_config_dict = None

    async def collective_rpc(self, method: str, args: tuple[Any, ...] = ()) -> Any:
        return await self.llm.collective_rpc(method, args=args)

    async def shutdown(self) -> bool:
        return await _shutdown_owner_llm_async(self)

    async def generate_and_score_async(
        self,
        prompts: list[str],
        sampling_params_kwargs: dict[str, Any],
        lora_request_specs: list[tuple[str, int, str]] | None,
        task_obj: Any,
        answers: list[Any],
        args: Any,
    ) -> tuple[list[float], dict[str, float], list[str]]:
        if hasattr(task_obj, "generate_and_score_async"):
            return await task_obj.generate_and_score_async(
                llm=self.llm,
                prompts=prompts,
                sampling_params_kwargs=sampling_params_kwargs,
                lora_request_specs=lora_request_specs,
                answers=answers,
                args=args,
            )

        # Default async bridge for non-agentic tasks (Math, Countdown, etc.)
        import asyncio
        import uuid

        sampling_params_obj = sampling_params(sampling_params_kwargs)
        lora_reqs = lora_requests(lora_request_specs) if lora_request_specs else None

        async def run_one(i: int):
            request_id = str(uuid.uuid4())
            lora_req = lora_reqs[i] if lora_reqs else None
            final_output = None
            async for out in self.llm.generate(prompts[i], sampling_params_obj, request_id, lora_request=lora_req):
                final_output = out
            return final_output

        tasks = [run_one(i) for i in range(len(prompts))]
        request_outputs = await asyncio.gather(*tasks)

        # Use the standard scoring logic (synchronous)
        return score_request_outputs(
            request_outputs,
            prompts=prompts,
            task_obj=task_obj,
            answers=answers,
            pass_at_k=bool(args.pass_at_k),
        )


async def _shutdown_owner_llm_async(owner: Any) -> bool:
    llm = getattr(owner, "llm", None)
    if llm is None:
        return True

    shutdown_fn = getattr(llm, "shutdown", None)
    if callable(shutdown_fn):
        try:
            res = shutdown_fn()
            import inspect

            if inspect.isawaitable(res):
                await res
        except Exception:
            pass

    _drop_llm(owner)
    return True


def _shutdown_owner_llm(owner: Any) -> bool:
    llm = getattr(owner, "llm", None)
    if llm is None:
        return True
    ok = _shutdown_llm(llm)
    _drop_llm(owner)
    return ok


def _shutdown_llm(llm: Any) -> bool:
    shutdown_fn = getattr(llm, "shutdown", None)
    if callable(shutdown_fn):
        return _call_noarg(shutdown_fn)
    engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
    return _shutdown_engine(engine)


def _shutdown_engine(engine: Any) -> bool:
    if engine is None:
        return True
    ok = True
    for name in ("shutdown", "shutdown_background_loop"):
        fn = getattr(engine, name, None)
        if callable(fn):
            ok = _call_noarg(fn) and ok
    return ok


def _call_noarg(fn: Any) -> bool:
    try:
        fn()
    except Exception:
        return False
    return True


def _drop_llm(owner: Any) -> None:
    try:
        del owner.llm
    except Exception:
        pass
    gc.collect()
