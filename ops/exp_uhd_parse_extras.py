from __future__ import annotations

from typing import Any

from ops.uhd_config import UHDConfig

_OPTIONAL_CFG_KEYS = (
    "steps_per_episode",
    "num_envs",
    "deterministic_policy",
    "seed_offset",
    "num_reps",
    "pretrain_search_dim",
    "pretrain_delta_scale",
    "pretrain_generation_length",
    "pretrain_rwkv_type",
    "pretrain_lora_only",
    "pretrain_basis_max_leaves",
    "max_tokens",
    "temperature",
    "samples_per_prompt",
    "prompt_batch_size",
    "pass_at_k",
    "num_gpus",
    "num_engines",
    "tensor_parallel_size",
    "sub_dataset_size",
    "hf_home",
    "text_search_dim",
    "text_delta_scale",
    "text_basis_max_tensors",
    "text_score_mode",
    "bf8_storage",
    "eggroll_noiser",
    "eggroll_rank",
    "eggroll_group_size",
    "eggroll_freeze_nonlora",
    "use_async",
    "vllm_enforce_eager",
    "vllm_max_model_len",
    "vllm_gpu_memory_utilization",
    "vllm_max_num_seqs",
    "vllm_max_num_batched_tokens",
    "vllm_speculative_method",
    "vllm_speculative_model",
    "vllm_num_speculative_tokens",
    "distill_teacher_model_choice",
    "distill_student_model_choice",
    "distill_dtype",
    "distill_generation_length",
    "distill_search_dim",
    "distill_delta_scale",
    "distill_lora_only",
    "distill_basis_max_leaves",
)


def apply_optional_cfg_fields(config_dict: dict[str, Any], cfg: dict[str, Any]) -> None:
    for key in _OPTIONAL_CFG_KEYS:
        if key in cfg:
            val = cfg[key]
            if key == "pretrain_basis_max_leaves" and val == 0:
                val = None
            config_dict[key] = val


def validate_llm_sampling_config(cfg: UHDConfig) -> None:
    if not str(cfg.env_tag).startswith("llm:"):
        return
    text_score_mode = str(cfg.text_score_mode)
    if text_score_mode not in {"generation", "nll"}:
        raise ValueError("UHD text configs require text_score_mode='generation' or text_score_mode='nll'.")
    if text_score_mode == "nll":
        if int(cfg.samples_per_prompt) != 1 or bool(cfg.pass_at_k):
            raise ValueError("UHD text NLL scoring requires samples_per_prompt=1 and pass_at_k=false.")
        return
    if int(cfg.samples_per_prompt) > 1 and float(cfg.temperature) <= 0.0:
        raise ValueError(
            "UHD text configs with samples_per_prompt > 1 require temperature > 0. "
            "vLLM greedy decoding uses temperature=0 and only supports n=1. "
            f"Got samples_per_prompt={cfg.samples_per_prompt}, temperature={cfg.temperature}."
        )
    if bool(cfg.pass_at_k) and int(cfg.samples_per_prompt) <= 1:
        raise ValueError("UHD text configs with pass_at_k=true require samples_per_prompt > 1.")
