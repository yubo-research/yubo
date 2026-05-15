from __future__ import annotations

from dataclasses import dataclass

from llm.registry import LLMEnvSpec, LLMPolicySpec


@dataclass(frozen=True)
class LLMConfig:
    env_tag: str
    policy_tag: str
    optimizer: str
    num_rounds: int | None
    total_timesteps: int | None
    num_epochs: int | None
    num_reps: int
    exp_dir: str
    log_file: str
    dry_run: bool
    hf_home: str | None
    problem_seed: int | None
    noise_seed_0: int | None
    seed_offset: int
    lr: float
    sigma: float
    log_interval: int
    target_accuracy: float | None
    batch_size: int
    population_size: int
    max_tokens: int
    temperature: float
    samples_per_prompt: int
    prompt_batch_size: int
    pass_at_k: bool
    normalize_with_std: bool
    scale_lr_in_grad: bool
    steps_per_adapter: int
    num_gpus: int | None
    num_engines: int | None
    tensor_parallel_size: int | None
    steps_per_eval: int
    eval_batch_size: int
    use_wandb: bool
    wandb_project: str
    wandb_name: str | None
    save_freq: int | None
    checkpoint_dir: str | None
    resume_from: str | None
    sub_dataset_size: int | None
    kl_beta: float | None
    reference_policy_tag: str | None
    env: LLMEnvSpec
    policy: LLMPolicySpec
    pretrain_lora_only: bool = True
    pretrain_search_dim: int = 4096
    vllm_max_model_len: int | None = None
    vllm_gpu_memory_utilization: float | None = None
    vllm_max_num_seqs: int | None = None
    vllm_max_num_batched_tokens: int | None = None


__all__ = ["LLMConfig"]
