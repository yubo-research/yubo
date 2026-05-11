from __future__ import annotations

import gc
import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm.lora import materialize_lora_adapters
from llm.vllm_scoring import score_request_outputs


def _external_import(*parts: str):
    return importlib.import_module("".join(parts))


@dataclass(frozen=True)
class VLLMActorConfig:
    model_name: str
    tensor_parallel_size: int
    max_loras: int
    lora_rank: int
    max_tokens: int
    prompt_batch_size: int
    enforce_eager: bool

    @classmethod
    def from_kwargs(cls, **kwargs) -> "VLLMActorConfig":
        return cls(**kwargs)


class EggrollVLLMActor:
    def __init__(self, config: VLLMActorConfig | None = None, **kwargs) -> None:
        cfg = config if config is not None else VLLMActorConfig.from_kwargs(**kwargs)
        _set_vllm_env_defaults()

        self.llm = _external_import("v", "llm").LLM(**_llm_kwargs(cfg))
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
        request_outputs = self.llm.generate(
            prompts,
            _sampling_params(sampling_params_kwargs),
            lora_request=_lora_requests(lora_request_specs),
            use_tqdm=True,
        )
        return score_request_outputs(request_outputs, prompts=prompts, task_obj=task_obj, answers=answers, pass_at_k=bool(args.pass_at_k))


class TextVLLMActor(EggrollVLLMActor):
    pass


def _set_vllm_env_defaults() -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_FUSED_MOE_CHUNK_SIZE", str(16 * 2048))
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _llm_kwargs(cfg: VLLMActorConfig) -> dict[str, Any]:
    return {
        "model": cfg.model_name,
        "tensor_parallel_size": int(cfg.tensor_parallel_size),
        "distributed_executor_backend": "ray",
        "worker_extension_cls": "llm.vllm_worker.WorkerExtension",
        "dtype": "auto",
        "enable_prefix_caching": True,
        "enforce_eager": bool(cfg.enforce_eager),
        "enable_lora": True,
        "max_loras": int(cfg.max_loras),
        "max_lora_rank": max(int(cfg.lora_rank), 8),
        "gpu_memory_utilization": 0.90,
        "trust_remote_code": True,
        "max_num_seqs": 512,
        "max_model_len": max(1024, 512 + int(cfg.max_tokens)),
        "max_num_batched_tokens": max(1, int(cfg.prompt_batch_size)) * 2048,
        "enable_chunked_prefill": True,
        "load_format": "auto",
    }


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


def _sampling_params(kwargs: dict[str, Any]):
    return _external_import("v", "llm").SamplingParams(**kwargs)


def _lora_requests(specs: list[tuple[str, int, str]] | None):
    if specs is None:
        return None

    LoRARequest = _external_import("v", "llm", ".lora", ".request").LoRARequest
    return [LoRARequest(lora_name=name, lora_int_id=int(lora_id), lora_path=path) for name, lora_id, path in specs]
