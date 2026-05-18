"""Configuration types for UHD experiments."""

from dataclasses import dataclass

from ops.uhd_config_types import BEConfig, EarlyRejectConfig, ENNConfig


@dataclass(frozen=True)
class UHDConfig:
    env_tag: str
    policy_tag: str | None
    num_rounds: int
    problem_seed: int | None
    noise_seed_0: int | None
    lr: float
    num_dim_target: float | None
    num_module_target: float | None
    log_interval: int
    accuracy_interval: int
    target_accuracy: float | None
    optimizer: str
    batch_size: int
    early_reject: EarlyRejectConfig
    be: BEConfig
    enn: ENNConfig
    bszo_k: int
    bszo_epsilon: float
    bszo_sigma_p_sq: float
    bszo_sigma_e_sq: float
    bszo_alpha: float
    sigma: float = 0.001
    steps_per_episode: int = 200
    num_envs: int = 1
    deterministic_policy: bool = False
    seed_offset: int = 0
    num_reps: int = 1
    total_timesteps: int | None = None
    pretrain_search_dim: int = 4096
    pretrain_delta_scale: float = 1.0
    pretrain_generation_length: int | None = None
    pretrain_rwkv_type: str | None = None
    pretrain_lora_only: bool = True
    pretrain_basis_max_leaves: int | None = 32
    max_tokens: int = 1024
    temperature: float = 0.0
    samples_per_prompt: int = 1
    prompt_batch_size: int = 2
    pass_at_k: bool = False
    num_gpus: int | None = None
    num_engines: int | None = None
    tensor_parallel_size: int | None = None
    sub_dataset_size: int | None = None
    hf_home: str | None = None
    text_search_dim: int = 256
    text_delta_scale: float = 1.0
    text_basis_max_tensors: int | None = 32
    bf8_storage: bool = False
    perturb_backend: str = "flat"
    eggroll_noiser: str = "eggroll"
    eggroll_rank: int = 1
    eggroll_group_size: int = 0
    eggroll_freeze_nonlora: bool = False
    use_async: bool = False
    vllm_max_model_len: int | None = None
    vllm_gpu_memory_utilization: float | None = None
    vllm_max_num_seqs: int | None = None
    vllm_max_num_batched_tokens: int | None = None
    vllm_speculative_method: str | None = None
    vllm_speculative_model: str | None = None
    vllm_num_speculative_tokens: int | None = None


__all__ = ["BEConfig", "EarlyRejectConfig", "ENNConfig", "UHDConfig"]
