from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any


def external_import(*parts: str):
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
    vllm_max_model_len: int | None = None

    @classmethod
    def from_kwargs(cls, **kwargs) -> "VLLMActorConfig":
        return cls(**kwargs)


def set_vllm_env_defaults() -> None:
    # Unset VLLM_PLUGINS means "load all discovered plugins"; the remote
    # torchrl plugin currently registers a broken Qwen3 override.
    os.environ.setdefault("VLLM_PLUGINS", "")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_FUSED_MOE_CHUNK_SIZE", str(16 * 2048))
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def get_llm_kwargs(cfg: VLLMActorConfig) -> dict[str, Any]:
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
        "max_model_len": int(cfg.vllm_max_model_len) if cfg.vllm_max_model_len is not None else max(8192, 512 + int(cfg.max_tokens)),
        "max_num_batched_tokens": max(1, int(cfg.prompt_batch_size)) * 2048,
        "enable_chunked_prefill": True,
        "load_format": "auto",
    }


def sampling_params(kwargs: dict[str, Any]):
    return external_import("v", "llm").SamplingParams(**kwargs)


def lora_requests(specs: list[tuple[str, int, str]] | None):
    if specs is None:
        return None

    LoRARequest = external_import("v", "llm", ".lora", ".request").LoRARequest
    return [LoRARequest(lora_name=name, lora_int_id=int(lora_id), lora_path=path) for name, lora_id, path in specs]


__all__ = [
    "VLLMActorConfig",
    "set_vllm_env_defaults",
    "get_llm_kwargs",
    "sampling_params",
    "lora_requests",
    "external_import",
]
