from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnginePoolConfig:
    model_name: str
    tensor_parallel_size: int
    num_engines: int
    lora_rank: int
    max_loras_per_engine: int
    max_tokens: int
    prompt_batch_size: int


class VLLMEnginePool:
    def __init__(self, *, ray: Any, engines: list[Any], placement_groups: list[Any]) -> None:
        self._ray = ray
        self.engines = engines
        self._placement_groups = placement_groups

    @classmethod
    def launch(cls, cfg: EnginePoolConfig) -> "VLLMEnginePool":
        import ray
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        from llm.vllm import TextVLLMActor

        ensure_ray(ray)
        placement_groups = []
        strategies = []
        for _ in range(int(cfg.num_engines)):
            bundles = [{"GPU": 1, "CPU": 2} for _ in range(int(cfg.tensor_parallel_size))]
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

        enforce_eager = int(cfg.tensor_parallel_size) > 1
        actors = [
            ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(TextVLLMActor).remote(
                model_name=str(cfg.model_name),
                tensor_parallel_size=int(cfg.tensor_parallel_size),
                max_loras=int(cfg.max_loras_per_engine),
                lora_rank=int(cfg.lora_rank),
                max_tokens=int(cfg.max_tokens),
                prompt_batch_size=int(cfg.prompt_batch_size),
                enforce_eager=enforce_eager,
            )
            for strategy in strategies
        ]
        pool = cls(ray=ray, engines=actors, placement_groups=placement_groups)
        pool.init_worker_groups()
        return pool

    def init_worker_groups(self) -> None:
        if not self.engines:
            raise RuntimeError("Cannot initialize an empty vLLM engine pool.")
        master_info = self._ray.get(self.engines[0].collective_rpc.remote("get_transport_info", args=()))[0]
        master_address, master_port = master_info
        results = self._ray.get(
            [
                self.engines[i].collective_rpc.remote("init_inter_engine_group", args=(master_address, master_port, i, len(self.engines)))
                for i in range(len(self.engines))
            ]
        )
        if not all(result[0] for result in results):
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

        refs = []
        requests_per_engine = max(1, -(-len(prompts) // len(self.engines)))
        for i, engine in enumerate(self.engines):
            start = i * requests_per_engine
            stop = min(start + requests_per_engine, len(prompts))
            if start >= stop:
                continue
            refs.append(
                engine.generate_and_score.remote(
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
            self._ray.wait(terminate_refs, timeout=30, num_returns=len(terminate_refs))
        except Exception:
            for engine in self.engines:
                try:
                    self._ray.kill(engine, no_restart=True)
                except Exception:
                    pass
        for pg in self._placement_groups:
            try:
                self._ray.util.remove_placement_group(pg)
            except Exception:
                pass


def ensure_ray(ray: Any) -> None:
    if ray.is_initialized():
        return
    try:
        ray.init(address="auto", include_dashboard=False, ignore_reinit_error=True)
    except Exception:
        ray.init(include_dashboard=False, ignore_reinit_error=True)


def sampling_kwargs(*, tokenizer: Any, temperature: float, seed: int, max_tokens: int, n: int) -> dict[str, Any]:
    stops = [token for token in (getattr(tokenizer, "eos_token", None), "<|im_end|>", "<|endoftext|>") if token]
    return {
        "temperature": float(temperature),
        "seed": int(seed),
        "max_tokens": int(max_tokens),
        "n": int(n),
        "stop": stops,
    }


__all__ = ["EnginePoolConfig", "VLLMEnginePool", "ensure_ray", "sampling_kwargs"]
