# Semantic LLM Update Programs

This note defines the first implementation boundary for architecture-aware Bayesian optimization over LLM update programs.

## Goal

The optimizer should not search raw parameter names such as `model.layers.12.mlp.down_proj.weight`. That representation is brittle across dense Transformers, sparse MoEs, and hybrid RNN/SSM models. The optimizer should search a small semantic program:

```text
roles = {attention_q, attention_v, mlp_down}
layer_band = middle
expert_policy = dense
rank = 4
scale = 0.03
seed = 123
```

An architecture adapter resolves this program into concrete tensors for the active model, and the runtime backend lowers those tensors to CUDA/vLLM or Metal/vLLM-Metal update paths.

## Paper Grounding

- LoRA shows that low-rank adaptation can move most of the useful task-specific update into small trainable matrices while freezing the base model: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", arXiv:2106.09685.
- Switch Transformer and Mixtral show that MoE models split the model into routers and sparsely activated experts, so expert parameters cannot be treated as ordinary dense MLP parameters: Fedus et al., arXiv:2101.03961; Jiang et al., arXiv:2401.04088.
- DeepSeekMoE adds shared and routed expert specialization, which is why the role vocabulary distinguishes `moe_shared_*` from `moe_expert_*`: Dai et al., arXiv:2401.06066.
- Mamba, RWKV, and state-space duality motivate explicit SSM/RNN mixer roles instead of assuming every architecture exposes attention and MLP blocks: Gu and Dao, arXiv:2312.00752; Peng et al., arXiv:2305.13048; Dao and Gu, arXiv:2405.21060.
- CODA motivates the constrained-program interface: keep the program space small enough that it can be lowered efficiently to the hardware backend instead of letting each model path become arbitrary glue code: Guo et al., "CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs", arXiv:2605.19269.

## Current Contract

`llm.architecture` provides:

- `discover_architecture_profile(model)`: traverse a Torch-like module tree and classify parameter tensors into semantic roles.
- `LLMUpdateProgram`: an immutable BO candidate description over roles, layer bands, expert policy, rank, scale, and seed.
- `resolve_update_program(profile, program)`: resolve the semantic candidate to concrete tensors or fail loudly.
- `update_program_features(profile, program)`: produce a stable numeric representation suitable for ENN/TuRBO-style acquisition.
- `lora_target_module_names(profile)`: list concrete module names that can be used by PEFT-style LoRA setup.

The implementation deliberately fails when a program matches no targets. Silent fallback would make cross-architecture results scientifically meaningless.

`llm.lora.build_peft_lora_template()` now uses this semantic discovery path when it has an instantiated empty model tree. That means dense Transformer, MoE expert/shared-expert, and SSM/RNN-style linear mixer modules are selected by role rather than by a fixed Qwen-style suffix list. The old suffix list remains only as the fallback for callers that request target modules before a model tree exists.

Both direct `[llm]` runs and `[uhd]` text runs accept the same target-selection fields:

```toml
llm_update_roles = ["attention_q", "mlp_down"]
llm_update_layer_band = "middle"
llm_update_expert_policy = "dense"
llm_update_max_targets = 16
```

Roles may also be written as a comma-separated string, for example `llm_update_roles = "moe_router,moe_expert_down"`.

For GPT-NeoX/Pythia-style models, fused `query_key_value` projections are represented by the `attention_qkv` role. This is intentionally distinct from separate `attention_q`, `attention_k`, and `attention_v` roles because the backend lowering is different. Adapter-materialized PEFT evaluation can target `attention_qkv`; direct in-place vLLM dense updates need an explicit fused-QKV lowerer before they should be marked supported.

## Inspection CLI

Use the inspection CLI before launching an expensive LLM run:

```bash
.pixi/envs/yubo/bin/python ops/llm_architecture.py inspect qwen3-1p7b-lora-r1
.pixi/envs/yubo/bin/python ops/llm_architecture.py inspect qwen3-1p7b-lora-r1 --roles attention_q,mlp_down --layer-band middle
.pixi/envs/yubo/bin/python ops/llm_architecture.py inspect pythia-14m-lora-r1 --roles attention_qkv,attention_o,mlp_down --layer-band middle --max-targets 6 --no-direct-vllm-dense-update
.pixi/envs/yubo/bin/python ops/llm_architecture.py inspect-config configs/uhd/text/gsm8k_qwen3_1p7b_mezo_be_enn.toml
```

The command builds an empty Hugging Face model under `accelerate.init_empty_weights()`, so it does not allocate full model weights or launch vLLM. It still needs the model config/code to be available from Hugging Face or local cache. Use `--local-files-only` when you want to prove the command is cache-only.

Use `--format json` for scripts and `--limit N` to control how many selected modules are printed.

## Backend Boundary

This change does not rewrite vLLM or vLLM-Metal kernels. It creates the architecture-neutral layer that those lowerings need.

There are two different execution paths:

- Adapter-materialized evaluation can use the broader PEFT target set selected by the semantic program.
- Direct in-place vLLM ES updates currently support only attention q/k/v/o, dense MLP gate/up/down, shared experts, and routed expert gate/up/down. Router-only, SSM, RWKV, and other targets fail early unless a backend lowerer is added.

The next backend step is to lower additional resolved update programs into the existing LoRA/subspace update paths while preserving:

- same BO semantics on CUDA and Metal,
- no extra CPU/GPU synchronization in the candidate loop,
- explicit unsupported-role errors for architectures the backend cannot update,
- timing telemetry for proposal, update application, and scoring.
