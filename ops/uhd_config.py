"""Configuration types for UHD experiments."""

from dataclasses import dataclass

from llm.architecture import DEFAULT_LORA_ROLES
from ops.uhd_config_types import BEConfig, EarlyRejectConfig, ENNConfig


@dataclass(frozen=True)
class UHDConfig:
    env_tag: str
    num_rounds: int
    problem_seed: int | None = None
    noise_seed_0: int | None = None
    lr: float = 0.001
    sigma: float = 0.001
    num_dim_target: float | None = None
    num_module_target: float | None = None
    log_interval: int = 1
    accuracy_interval: int = 1000
    target_accuracy: float | None = None
    optimizer: str = "mezo"
    batch_size: int = 4096
    early_reject: EarlyRejectConfig = EarlyRejectConfig(None, None, None, None, None, None)
    be: BEConfig = BEConfig(10, 10, 20, 10, 25, None)
    enn: ENNConfig = ENNConfig(False, 100, 4, 123, 25, 50, 200, 50, 0.25, "mu_minus", 1, 1, "direction", 64)
    bszo_k: int = 2
    bszo_epsilon: float = 1e-4
    bszo_sigma_p_sq: float = 1.0
    bszo_sigma_e_sq: float = 1.0
    bszo_alpha: float = 0.1
    policy_tag: str | None = None
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
    text_score_mode: str = "generation"
    llm_update_roles: tuple[str, ...] = DEFAULT_LORA_ROLES
    llm_update_layer_band: str = "all"
    llm_update_expert_policy: str = "all"
    llm_update_max_targets: int | None = None
    bf8_storage: bool = False
    perturb_backend: str = "flat"
    eggroll_noiser: str = "eggroll"
    eggroll_rank: int = 1
    eggroll_group_size: int = 0
    eggroll_freeze_nonlora: bool = False
    use_async: bool = False
    vllm_enforce_eager: bool = False
    vllm_max_model_len: int | None = None
    vllm_gpu_memory_utilization: float | None = None
    vllm_max_num_seqs: int | None = None
    vllm_max_num_batched_tokens: int | None = None
    vllm_speculative_method: str | None = None
    vllm_speculative_model: str | None = None
    vllm_num_speculative_tokens: int | None = None
    distill_teacher_model_choice: str | None = None
    distill_student_model_choice: str | None = None
    distill_dtype: str | None = None
    distill_generation_length: int | None = None
    distill_search_dim: int | None = None
    distill_delta_scale: float | None = None
    distill_lora_only: bool | None = None
    distill_basis_max_leaves: int | None = None


__all__ = ["BEConfig", "EarlyRejectConfig", "ENNConfig", "UHDConfig"]
