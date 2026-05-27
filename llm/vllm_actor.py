from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import numpy as np

from llm.lora import materialize_lora_adapters
from llm.model_client import SampleBatch, SampleCall
from llm.tasks_base import is_rollout_task
from llm.universal_subspace import (
    universal_global_pop_pair_idx,
    universal_subspace_seed,
)
from llm.vllm_actor_config import (
    VLLMActorConfig,
    external_import,
    get_llm_kwargs,
    lora_requests,
    sampling_params,
    set_vllm_env_defaults,
)
from llm.vllm_model_client import VLLMSampler
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
        self._universal_template = None
        self._universal_template_uploaded = False

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

    def set_engine_rank(self, rank: int) -> bool:
        self.rank = int(rank)
        return True

    def set_universal_subspace_template(self, template: Any) -> bool:
        # Cache locally for seed math and push once into vLLM workers so we can
        # perturb/unperturb without repeatedly serializing the template.
        print(f"ACTOR: Uploading universal template (search_dim={getattr(template, 'search_dim', 'unknown')})...")
        self._universal_template = template
        self.llm.collective_rpc("set_universal_subspace_template", args=(template,))
        self._universal_template_uploaded = True
        print("ACTOR: Universal template upload complete.")
        return True

    def get_parameter_metadata(self) -> Any:
        """Discovers parameters from the internal vLLM workers."""
        return self.llm.collective_rpc("discover_parameters")

    def apply_universal_update(
        self,
        normalized_fitnesses: list[float],
        template_ref: Any | None,
        es_step: int,
        args: Any,
    ) -> bool:
        """Applies a universal coordinate update across all workers."""
        template = template_ref
        if template is None:
            template = self._universal_template
        if template is None:
            raise RuntimeError("Universal subspace template not set on actor.")
        if not self._universal_template_uploaded or template is not self._universal_template:
            # Fallback: still works, but may be slower because the template is
            # serialized to worker processes on each call path.
            self.llm.collective_rpc(
                "apply_universal_es_update",
                args=(normalized_fitnesses, template, es_step, args),
            )
            return True
        return self.llm.collective_rpc(
            "apply_universal_es_update",
            args=(normalized_fitnesses, None, es_step, args),
        )

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

    def sample(self, calls: list[SampleCall]) -> list[SampleBatch]:
        return _run_blocking(_sample_many(self.llm, calls))

    def generate(self, requests: list[SampleCall]) -> list[SampleBatch]:
        return self.sample(requests)

    def generate_and_score(
        self,
        prompts: list[str],
        sampling_params_kwargs: dict[str, Any],
        lora_request_specs: list[tuple[str, int, str]] | None,
        task_obj: Any,
        answers: list[Any],
        args: Any,
        *,
        es_step: int | None = None,
        subspace_template_ref: Any | None = None,
    ) -> tuple[list[float], dict[str, float], list[str]]:
        pretrain_lora_only = bool(getattr(args, "pretrain_lora_only", True))
        if subspace_template_ref is not None or not pretrain_lora_only:
            return self._generate_and_score_universal(
                prompts=prompts,
                sampling_params_kwargs=sampling_params_kwargs,
                template=subspace_template_ref,
                task_obj=task_obj,
                answers=answers,
                args=args,
                es_step=es_step,
            )

        if is_rollout_task(task_obj):
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
            use_tqdm=False,
        )
        return score_request_outputs(
            request_outputs,
            prompts=prompts,
            task_obj=task_obj,
            answers=answers,
            pass_at_k=bool(args.pass_at_k),
        )

    def _generate_and_score_universal(
        self,
        prompts: list[str],
        sampling_params_kwargs: dict[str, Any],
        template: Any,
        task_obj: Any,
        answers: list[Any],
        args: Any,
        es_step: int | None,
    ) -> tuple[list[float], dict[str, float], list[str]]:
        """Handles sequential perturbation and generation for universal subspace arms."""
        if template is None:
            template = self._universal_template
        if template is None:
            raise RuntimeError("Universal subspace template not set on actor.")

        population_size_per_engine = int(args.population_size) // int(args.num_engines)
        num_pop_pairs = population_size_per_engine // 2
        global_num_pop_pairs = int(args.population_size) // 2

        all_fitnesses = []
        all_info_lists: dict[str, list[float]] = {}
        all_logs = []

        # Keep this in lock-step with apply_universal_es_update() so seeds match
        # the update step.
        pop_step = int(es_step or 0) // max(1, int(args.steps_per_adapter))
        total_arms = num_pop_pairs * 2
        arm_idx = 0

        for pair_idx in range(num_pop_pairs):
            global_pair_idx = universal_global_pop_pair_idx(
                engine_rank=int(self.rank),
                num_engines=int(args.num_engines),
                population_size=int(args.population_size),
                local_pop_pair_idx=pair_idx,
            )
            for sign in [1.0, -1.0]:
                arm_idx += 1
                if arm_idx % 10 == 1 or arm_idx == total_arms:
                    print(f"ACTOR: Universal Arm {arm_idx}/{total_arms} (Pair {pair_idx + 1}, Sign {sign})")

                # 1. Perturb
                seed = universal_subspace_seed(
                    base_seed=int(args.base_seed),
                    num_pop_pairs=global_num_pop_pairs,
                    search_dim=int(template.search_dim),
                    pop_step=pop_step,
                    pop_pair_idx=global_pair_idx,
                )
                # If we've already uploaded the template into the worker
                # processes, avoid re-sending it on each perturbation call.
                template_arg = None if self._universal_template_uploaded and template is self._universal_template else template
                self.llm.collective_rpc(
                    "apply_subspace_perturbation",
                    args=(template_arg, seed, float(args.sigma) * sign),
                )

                # 2. Generate
                request_outputs = self.llm.generate(
                    prompts,
                    sampling_params(sampling_params_kwargs),
                    use_tqdm=False,
                )

                # 3. Score
                fitnesses, info, logs = score_request_outputs(
                    request_outputs,
                    prompts=prompts,
                    task_obj=task_obj,
                    answers=answers,
                    pass_at_k=bool(args.pass_at_k),
                )
                all_fitnesses.extend(fitnesses)
                all_logs.extend(logs)
                for k, v in info.items():
                    all_info_lists.setdefault(k, []).append(v)

                # 4. Unperturb
                self.llm.collective_rpc(
                    "apply_subspace_perturbation",
                    args=(template_arg, seed, -float(args.sigma) * sign),
                )

        # Aggregate info
        final_info = {k: float(np.mean(v)) for k, v in all_info_lists.items()}
        return all_fitnesses, final_info, all_logs


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
        self._universal_template = None
        self._universal_template_uploaded = False

    async def collective_rpc(self, method: str, args: tuple[Any, ...] = ()) -> Any:
        return await self.llm.collective_rpc(method, args=args)

    async def shutdown(self) -> bool:
        return await _shutdown_owner_llm_async(self)

    async def sample(self, calls: list[SampleCall]) -> list[SampleBatch]:
        return await _sample_many(self.llm, calls)

    async def generate(self, requests: list[SampleCall]) -> list[SampleBatch]:
        return await self.sample(requests)

    async def generate_and_score_async(
        self,
        prompts: list[str],
        sampling_params_kwargs: dict[str, Any],
        lora_request_specs: list[tuple[str, int, str]] | None,
        task_obj: Any,
        answers: list[Any],
        args: Any,
    ) -> tuple[list[float], dict[str, float], list[str]]:
        if is_rollout_task(task_obj):
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


async def _sample_many(llm: Any, calls: list[SampleCall]) -> list[SampleBatch]:
    sampler = VLLMSampler(llm)
    import asyncio

    return await asyncio.gather(*(sampler.sample(call) for call in calls))


def _run_blocking(awaitable: Any) -> Any:
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError("Cannot run synchronous vLLM actor generation from an active event loop.")


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
