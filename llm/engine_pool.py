from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from llm.model_client import AdapterRef, SampleBatch, SampleCall
from llm.ray_cleanup import cleanup_ray_launch, kill_ray_actors
from llm.tasks_base import is_rollout_task
from llm.vllm_scoring import score_completions


@dataclass(frozen=True)
class EnginePoolConfig:
    model_name: str
    tensor_parallel_size: int
    num_engines: int
    lora_rank: int
    max_loras_per_engine: int
    max_tokens: int
    prompt_batch_size: int
    samples_per_prompt: int = 1
    use_async: bool = False
    vllm_enforce_eager: bool = False
    vllm_max_model_len: int | None = None
    vllm_gpu_memory_utilization: float | None = None
    vllm_max_num_seqs: int | None = None
    vllm_max_num_batched_tokens: int | None = None
    vllm_speculative_method: str | None = None
    vllm_speculative_model: str | None = None
    vllm_num_speculative_tokens: int | None = None


def vllm_placement_bundles(ray: Any, tensor_parallel_size: int) -> list[dict[str, float | int]]:
    size = max(1, int(tensor_parallel_size))
    if float(ray.cluster_resources().get("GPU", 0)) > 0:
        return [{"GPU": 1, "CPU": 2} for _ in range(size)]
    return [{"CPU": 2} for _ in range(size)]


class VLLMEnginePool:
    def __init__(
        self,
        *,
        ray: Any,
        engines: list[Any],
        placement_groups: list[Any],
        use_async: bool = False,
    ) -> None:
        self._ray = ray
        self.engines = engines
        self._placement_groups = placement_groups
        self.use_async = bool(use_async)

    @classmethod
    def launch(cls, cfg: EnginePoolConfig) -> "VLLMEnginePool":
        import ray
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        from llm.vllm_actor import AsyncTextVLLMActor, TextVLLMActor

        ensure_ray(ray)
        placement_groups = []
        strategies = []
        actors = []
        try:
            for _ in range(int(cfg.num_engines)):
                bundles = vllm_placement_bundles(ray, int(cfg.tensor_parallel_size))
                pg = placement_group(bundles, lifetime="detached", strategy="STRICT_PACK")
                ray.get(pg.ready())
                placement_groups.append(pg)
                if int(cfg.tensor_parallel_size) == 1:
                    strategies.append(
                        PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_capture_child_tasks=True,
                            placement_group_bundle_index=0,
                        )
                    )
                else:
                    strategies.append(
                        PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_capture_child_tasks=True,
                        )
                    )

            enforce_eager = bool(cfg.vllm_enforce_eager) or int(cfg.tensor_parallel_size) > 1

            actor_cls = AsyncTextVLLMActor if cfg.use_async else TextVLLMActor
            concurrency = 1

            actors = [
                ray.remote(
                    num_cpus=0,
                    num_gpus=0,
                    scheduling_strategy=strategy,
                    max_concurrency=concurrency,
                    runtime_env=None,
                )(actor_cls).remote(
                    model_name=str(cfg.model_name),
                    tensor_parallel_size=int(cfg.tensor_parallel_size),
                    max_loras=int(cfg.max_loras_per_engine),
                    lora_rank=int(cfg.lora_rank),
                    max_tokens=int(cfg.max_tokens),
                    prompt_batch_size=int(cfg.prompt_batch_size),
                    samples_per_prompt=int(cfg.samples_per_prompt),
                    enforce_eager=enforce_eager,
                    vllm_max_model_len=cfg.vllm_max_model_len,
                    vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
                    vllm_max_num_seqs=cfg.vllm_max_num_seqs,
                    vllm_max_num_batched_tokens=cfg.vllm_max_num_batched_tokens,
                    vllm_speculative_method=cfg.vllm_speculative_method,
                    vllm_speculative_model=cfg.vllm_speculative_model,
                    vllm_num_speculative_tokens=cfg.vllm_num_speculative_tokens,
                )
                for strategy in strategies
            ]
            pool = cls(
                ray=ray,
                engines=actors,
                placement_groups=placement_groups,
                use_async=cfg.use_async,
            )
            pool.init_worker_groups()
            return pool
        except Exception:
            cleanup_ray_launch(ray, actors, placement_groups)
            raise

    def init_worker_groups(self) -> None:
        if not self.engines:
            raise RuntimeError("Cannot initialize an empty vLLM engine pool.")
        master_info = self._ray.get(self.engines[0].collective_rpc.remote("get_transport_info", args=()))
        master_by_tensor_rank = transport_info_by_tensor_rank(master_info)
        results = self._ray.get(
            [
                self.engines[i].collective_rpc.remote(
                    "init_inter_engine_group",
                    args=(master_by_tensor_rank, i, len(self.engines)),
                )
                for i in range(len(self.engines))
            ]
        )
        if not collective_results_ok(results):
            raise RuntimeError("Failed to initialize at least one vLLM worker group.")

    def generate_and_score(
        self,
        *,
        prompts: list[str],
        sampling_params_kwargs: dict[str, Any],
        lora_request_specs: list[tuple[str, int, str]] | None,
        task_obj: Any,
        answers: list[Any],
        args: Any,
    ) -> tuple[list[float], dict[str, float], list[str]]:
        if len(prompts) != len(answers):
            raise ValueError("prompts and answers must have the same length.")
        if lora_request_specs is not None and len(lora_request_specs) != len(prompts):
            raise ValueError("lora_request_specs must be None or have one entry per prompt.")
        if is_rollout_task(task_obj):
            return self._generate_and_score_in_actor(
                prompts=prompts,
                sampling_params_kwargs=sampling_params_kwargs,
                lora_request_specs=lora_request_specs,
                task_obj=task_obj,
                answers=answers,
                args=args,
            )

        responses = self.sample(
            _calls_from_prompts(
                prompts=prompts,
                sampling_params_kwargs=sampling_params_kwargs,
                lora_request_specs=lora_request_specs,
            )
        )
        return score_completions(
            responses,
            prompts=prompts,
            task_obj=task_obj,
            answers=answers,
            pass_at_k=bool(args.pass_at_k),
        )

    def sample(self, calls: list[SampleCall]) -> list[SampleBatch]:
        if not calls:
            return []
        refs = []
        calls_per_engine = max(1, -(-len(calls) // len(self.engines)))
        for i, engine in enumerate(self.engines):
            start = i * calls_per_engine
            stop = min(start + calls_per_engine, len(calls))
            if start >= stop:
                continue
            refs.append(engine.sample.remote(calls[start:stop]))
        results = self._ray.get(refs)
        batches: list[SampleBatch] = []
        for engine_batches in results:
            batches.extend(engine_batches)
        return batches

    def generate(self, requests: list[SampleCall]) -> list[SampleBatch]:
        return self.sample(requests)

    def _generate_and_score_in_actor(
        self,
        *,
        prompts: list[str],
        sampling_params_kwargs: dict[str, Any],
        lora_request_specs: list[tuple[str, int, str]] | None,
        task_obj: Any,
        answers: list[Any],
        args: Any,
    ) -> tuple[list[float], dict[str, float], list[str]]:
        refs = []
        requests_per_engine = max(1, -(-len(prompts) // len(self.engines)))
        for i, engine in enumerate(self.engines):
            start = i * requests_per_engine
            stop = min(start + requests_per_engine, len(prompts))
            if start >= stop:
                continue
            method = engine.generate_and_score_async if self.use_async else engine.generate_and_score
            refs.append(
                method.remote(
                    prompts[start:stop],
                    sampling_params_kwargs,
                    None if lora_request_specs is None else lora_request_specs[start:stop],
                    task_obj,
                    answers[start:stop],
                    args,
                )
            )

        results = self._ray.get(refs)
        fitnesses: list[float] = []
        info_by_key: dict[str, list[float]] = {}
        logs: list[str] = []
        for eng_fitness, info, eng_logs in results:
            fitnesses.extend(float(x) for x in eng_fitness)
            for key, value in info.items():
                info_by_key.setdefault(str(key), []).append(float(value))
            logs.extend(str(text) for text in eng_logs)
        info = {key: sum(values) / len(values) for key, values in info_by_key.items() if values}
        return fitnesses, info, logs

    def shutdown(self) -> None:
        try:
            self._ray.get([engine.shutdown.remote() for engine in self.engines], timeout=60)
        except Exception:
            pass
        try:
            terminate_refs = [engine.__ray_terminate__.remote() for engine in self.engines]
            _, pending = self._ray.wait(terminate_refs, timeout=30, num_returns=len(terminate_refs))
            if pending:
                kill_ray_actors(self._ray, self.engines)
        except Exception:
            kill_ray_actors(self._ray, self.engines)
        for pg in self._placement_groups:
            try:
                self._ray.util.remove_placement_group(pg)
            except Exception:
                pass


def ensure_ray(ray: Any) -> None:
    if ray.is_initialized():
        return
    _apply_ray_env_vars()
    try:
        ray.init(
            address="auto",
            include_dashboard=False,
            ignore_reinit_error=True,
            runtime_env=None,
        )
    except Exception:
        ray.init(include_dashboard=False, ignore_reinit_error=True, runtime_env=None)


def ray_runtime_env() -> dict[str, dict[str, str]] | None:
    env_vars = ray_env_vars()
    return {"env_vars": env_vars} if env_vars else None


def _apply_ray_env_vars() -> None:
    for key, value in ray_env_vars().items():
        os.environ.setdefault(key, value)


def ray_env_vars() -> dict[str, str]:
    env_vars: dict[str, str] = {}
    for key in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "PRIME_API_KEY",
        "VLLM_PLUGINS",
        "VLLM_METAL_MEMORY_FRACTION",
        "VLLM_MLX_DEVICE",
        "VLLM_DISTRIBUTED_EXECUTOR_BACKEND",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "THM_SANDBOX_BACKEND",
        "THM_DOCKER_RUNTIME",
        "THM_DOCKER_MEMORY",
        "THM_DOCKER_CPUS",
        "THM_DOCKER_ENTRYPOINT",
        "THM_DOCKER_USER",
        "THM_DOCKER_HOME",
        "THM_DOCKER_PATH_PREFIX",
        "THM_LEAN4_DOCKER_IMAGE",
        "THM_COQ_DOCKER_IMAGE",
        "THM_ISABELLE_DOCKER_IMAGE",
        "YUBO_LORA_POPULATION_PATH",
    ):
        value = os.environ.get(key)
        if value:
            env_vars[key] = str(value)
    return env_vars


def transport_info_by_tensor_rank(worker_infos: Any) -> dict[int, tuple[str, int]]:
    infos = worker_infos if isinstance(worker_infos, list) else [worker_infos]
    out: dict[int, tuple[str, int]] = {}
    for idx, info in enumerate(infos):
        if isinstance(info, dict):
            tensor_rank = int(info.get("tensor_rank", idx))
            host_value = info.get("host") or info.get("address")
            if not host_value:
                raise RuntimeError(f"vLLM transport info for tensor rank {tensor_rank} did not include a host.")
            host = str(host_value)
            port = int(info["port"])
        else:
            host, port = info
            tensor_rank = int(idx)
        out[tensor_rank] = (host, port)
    if not out:
        raise RuntimeError("vLLM collective_rpc returned no transport info.")
    return out


def collective_results_ok(results: Any) -> bool:
    def flatten(value: Any):
        if isinstance(value, (list, tuple)):
            for item in value:
                yield from flatten(item)
        else:
            yield value

    values = list(flatten(results))
    return bool(values) and all(bool(item) for item in values)


def _calls_from_prompts(
    *,
    prompts: list[str],
    sampling_params_kwargs: dict[str, Any],
    lora_request_specs: list[tuple[str, int, str]] | None,
) -> list[SampleCall]:
    calls: list[SampleCall] = []
    for idx, prompt in enumerate(prompts):
        adapter = AdapterRef.from_tuple(None if lora_request_specs is None else lora_request_specs[idx])
        calls.append(
            SampleCall(
                prompt=str(prompt),
                sampling=dict(sampling_params_kwargs),
                adapter=adapter,
            )
        )
    return calls


def sampling_kwargs(*, tokenizer: Any, temperature: float, seed: int, max_tokens: int, n: int) -> dict[str, Any]:
    stops = [
        token
        for token in (
            getattr(tokenizer, "eos_token", None),
            "<|im_end|>",
            "<|endoftext|>",
        )
        if token
    ]
    return {
        "temperature": float(temperature),
        "seed": int(seed),
        "max_tokens": int(max_tokens),
        "n": int(n),
        "stop": stops,
    }


__all__ = [
    "EnginePoolConfig",
    "VLLMEnginePool",
    "collective_results_ok",
    "ensure_ray",
    "ray_runtime_env",
    "sampling_kwargs",
    "transport_info_by_tensor_rank",
    "vllm_placement_bundles",
]
