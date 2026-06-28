from __future__ import annotations

import re
from typing import Any

from llm.architecture import LLMUpdateProgram, discover_architecture_profile, lora_target_module_names, resolve_update_program

LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_GEMMA4_LANGUAGE_LORA_TARGETS = (
    r".*language_model\.(model\.)?layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))(\.linear)?$"
)


def _lora_target_modules(model_name: str, base_model: Any | None = None, update_program: LLMUpdateProgram | None = None) -> list[str] | str:
    if base_model is None:
        if update_program is not None:
            raise ValueError("Architecture-aware LoRA target selection requires base_model.")
        if str(model_name).startswith("google/gemma-4-"):
            return _GEMMA4_LANGUAGE_LORA_TARGETS
        return list(LORA_TARGET_MODULES)
    if str(model_name).startswith("google/gemma-4-"):
        if update_program is None:
            return _discover_gemma4_lora_targets(base_model)
    return _discover_semantic_lora_targets(base_model, update_program=update_program)


def select_lora_target_modules(*, model_name: str, base_model: Any, update_program: LLMUpdateProgram | None = None) -> list[str]:
    targets = _lora_target_modules(model_name, base_model=base_model, update_program=update_program)
    if isinstance(targets, str):
        raise ValueError("LoRA target selection requires a concrete model tree, not a regex target pattern.")
    return targets


def validate_vllm_dense_update_support(peft_shapes_dict: dict[str, tuple[int, ...]]) -> None:
    unsupported = tuple(name for name in peft_shapes_dict if not _supports_vllm_dense_update(name))
    if unsupported:
        sample = ", ".join(unsupported[:5])
        raise ValueError(
            "Direct vLLM dense ES update does not support every selected LoRA target. "
            "Supported direct-update targets are attention q/k/v/o, dense MLP gate/up/down, shared experts, and routed expert gate/up/down. "
            "For router-only, fused-QKV, SSM, RWKV, or other targets, use adapter-materialized evaluation or add an explicit backend lowerer first. "
            f"Unsupported PEFT layer(s): {sample}"
        )


def unsupported_vllm_dense_update_modules(module_names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(module_name for module_name in module_names if not _supports_vllm_dense_update(f"base_model.model.{module_name}.base_layer.weight"))


def _supports_vllm_dense_update(peft_name: str) -> bool:
    name = str(peft_name).replace("base_model.model.", "")
    dense_markers = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "shared_expert.gate_proj",
        "shared_expert.up_proj",
        "shared_expert.down_proj",
    )
    if any(marker in name for marker in dense_markers):
        return True
    return bool(re.search(r"experts\.\d+\.(gate_proj|up_proj|down_proj)", name))


def _discover_semantic_lora_targets(base_model: Any, *, update_program: LLMUpdateProgram | None) -> list[str]:
    profile = discover_architecture_profile(base_model)
    if update_program is None:
        targets = lora_target_module_names(profile)
    else:
        targets = _module_names_from_targets(resolve_update_program(profile, update_program))
    targets = [target for target in targets if not _is_non_language_path(target)]
    if not targets:
        sample = [name for name, _module in list(base_model.named_modules())[:12]]
        raise ValueError(f"No semantic LoRA target modules discovered. Sample modules: {sample}")
    return targets


def _module_names_from_targets(targets: tuple[Any, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for target in targets:
        if target.parameter_name == "weight" and target.module_name not in names:
            names.append(target.module_name)
    return tuple(names)


def _discover_gemma4_lora_targets(base_model: Any) -> list[str]:
    targets = [name for name, module in base_model.named_modules() if _is_linear_module(module) and _is_gemma4_text_lora_name(str(name))]
    if not targets:
        sample = [name for name, _module in list(base_model.named_modules())[:12]]
        raise ValueError(f"No Gemma 4 text LoRA target modules discovered. Sample modules: {sample}")
    return targets


def _is_linear_module(module: Any) -> bool:
    return module.__class__.__name__ == "Linear"


def _is_gemma4_text_lora_name(name: str) -> bool:
    if _is_non_language_path(name):
        return False
    if not _is_lora_projection_leaf(name):
        return False
    return ".layers." in str(name)


def _is_non_language_path(name: str) -> bool:
    parts = str(name).split(".")
    return any(
        part
        in {
            "audio_model",
            "audio_tower",
            "image_tower",
            "multi_modal_projector",
            "visual",
            "vision_model",
            "vision_tower",
        }
        for part in parts
    )


def _is_lora_projection_leaf(name: str) -> bool:
    value = str(name)
    return any(value.endswith(f".{target}") or value.endswith(f".{target}.linear") for target in LORA_TARGET_MODULES)
