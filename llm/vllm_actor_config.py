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
    samples_per_prompt: int = 1
    vllm_max_model_len: int | None = None
    vllm_gpu_memory_utilization: float | None = None
    vllm_max_num_seqs: int | None = None
    vllm_max_num_batched_tokens: int | None = None

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
    max_model_len = _max_model_len(cfg)
    max_num_seqs = _max_num_seqs(cfg)
    return {
        "model": cfg.model_name,
        "tensor_parallel_size": int(cfg.tensor_parallel_size),
        "distributed_executor_backend": "ray",
        "worker_extension_cls": "llm.vllm_worker.WorkerExtension",
        "dtype": "auto",
        "enable_prefix_caching": False,
        "enforce_eager": bool(cfg.enforce_eager),
        "enable_lora": True,
        "max_loras": int(cfg.max_loras),
        "max_lora_rank": max(int(cfg.lora_rank), 8),
        "gpu_memory_utilization": _gpu_memory_utilization(cfg),
        "trust_remote_code": True,
        "max_num_seqs": max_num_seqs,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": _max_num_batched_tokens(cfg, max_model_len=max_model_len, max_num_seqs=max_num_seqs),
        "enable_chunked_prefill": True,
        "load_format": "auto",
    }


def _max_model_len(cfg: VLLMActorConfig) -> int:
    if cfg.vllm_max_model_len is not None:
        return max(1, int(cfg.vllm_max_model_len))
    return max(8192, 512 + int(cfg.max_tokens))


def _active_sequences(cfg: VLLMActorConfig) -> int:
    return max(1, int(cfg.prompt_batch_size)) * max(1, int(cfg.samples_per_prompt))


def _max_num_seqs(cfg: VLLMActorConfig) -> int:
    if cfg.vllm_max_num_seqs is not None:
        return max(1, int(cfg.vllm_max_num_seqs))
    return max(8, min(64, _active_sequences(cfg)))


def _gpu_memory_utilization(cfg: VLLMActorConfig) -> float:
    if cfg.vllm_gpu_memory_utilization is not None:
        return max(0.01, min(0.99, float(cfg.vllm_gpu_memory_utilization)))
    return 0.85


def _max_num_batched_tokens(cfg: VLLMActorConfig, *, max_model_len: int, max_num_seqs: int) -> int:
    if cfg.vllm_max_num_batched_tokens is not None:
        return max(1, int(cfg.vllm_max_num_batched_tokens))
    active = min(max_num_seqs, _active_sequences(cfg))
    return max(1, active) * min(max_model_len, 2048)


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
