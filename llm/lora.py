from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm.runtime_messages import missing_runtime_message


LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class LoraTemplate:
    state_dict: dict[str, Any]
    base_shapes: dict[str, tuple[int, ...]]
    config: dict[str, Any]


def build_peft_lora_template(*, model_name: str, rank: int, alpha: int) -> LoraTemplate:
    missing = [module for module in ("accelerate", "peft", "torch", "transformers") if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError(_missing_runtime_message(missing))
    try:
        import torch
        from accelerate import init_empty_weights
        from peft import LoraConfig, get_peft_model
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(_missing_runtime_message(["accelerate", "peft", "torch", "transformers"])) from exc

    lora_config = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        target_modules=list(LORA_TARGET_MODULES),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    with init_empty_weights():
        base_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    peft_model = get_peft_model(base_model, lora_config)

    state_dict: dict[str, Any] = {}
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            continue
        tensor = torch.zeros(tuple(param.shape), device="cpu", dtype=torch.float16)
        if "lora_A" in name:
            torch.nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
        state_dict[name] = tensor

    base_shapes = {name: tuple(int(dim) for dim in param.shape) for name, param in peft_model.named_parameters() if name.endswith(".base_layer.weight")}
    return LoraTemplate(
        state_dict=state_dict,
        base_shapes=base_shapes,
        config=_jsonable_lora_config(lora_config.to_dict()),
    )


def materialize_lora_adapters(
    *,
    adapter_root: str | os.PathLike[str],
    rank: int,
    population_indices: list[int],
    es_step: int,
    args: Any,
    peft_state_dict: dict[str, Any],
    peft_shapes_dict: dict[str, tuple[int, ...]],
    lora_config_dict: dict[str, Any],
) -> list[str]:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("LoRA adapter materialization requires torch and safetensors.") from exc

    root = Path(adapter_root).expanduser() / f"rank_{rank}"
    root.mkdir(parents=True, exist_ok=True)
    config_to_save = _jsonable_lora_config(copy.deepcopy(lora_config_dict))
    pop_step = int(es_step) // int(args.steps_per_adapter)
    adapter_paths: list[str] = []

    for pop_idx in population_indices:
        adapter_path = root / f"pop_{int(pop_idx)}"
        if adapter_path.exists():
            shutil.rmtree(adapter_path)
        adapter_path.mkdir(parents=True, exist_ok=True)
        with open(adapter_path / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(config_to_save, f)

        local_state_dict: dict[str, Any] = {}
        for layer_idx, (peft_name, weight_shape) in enumerate(peft_shapes_dict.items()):
            lora_a_raw = peft_name.replace("base_layer.weight", "lora_A.default.weight")
            lora_b_raw = peft_name.replace("base_layer.weight", "lora_B.default.weight")
            lora_a_name = lora_a_raw.replace(".lora_A.default.weight", ".lora_A.weight")
            lora_b_name = lora_b_raw.replace(".lora_B.default.weight", ".lora_B.weight")

            lora_a = peft_state_dict[lora_a_raw].clone().cpu()
            lora_b = peft_state_dict[lora_b_raw].clone().cpu()

            lora_b_shape = (int(weight_shape[0]), int(args.lora_r))
            lora_a_shape = (int(args.lora_r), int(weight_shape[1]))
            noise_a, noise_b = get_rng_noise(
                base_seed=int(args.base_seed),
                num_pop_pairs=int(args.population_size) // 2,
                pop_pair_idx=int(pop_idx) // 2,
                num_layers=len(peft_shapes_dict),
                layer_idx=layer_idx,
                step=pop_step,
                shapes=[lora_a_shape, lora_b_shape],
            )
            noise_a = noise_a * math.sqrt(float(args.sigma))
            noise_b = noise_b * math.sqrt(float(args.sigma) / float(args.lora_r))

            lora_a.zero_()
            lora_b.zero_()
            lora_a.add_(noise_a.to(dtype=lora_a.dtype))
            lora_b.add_(((-noise_b) if int(pop_idx) % 2 else noise_b).to(dtype=lora_b.dtype))

            local_state_dict[lora_a_name] = lora_a
            local_state_dict[lora_b_name] = lora_b

        save_file(local_state_dict, str(adapter_path / "adapter_model.safetensors"))
        adapter_paths.append(str(adapter_path))

    return adapter_paths


def get_rng_noise(
    *,
    base_seed: int,
    num_pop_pairs: int,
    pop_pair_idx: int,
    num_layers: int,
    layer_idx: int,
    step: int,
    shapes: list[tuple[int, ...]],
) -> tuple[Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("EggRoll LoRA noise generation requires torch.") from exc

    seed = int(base_seed) + int(num_pop_pairs) * int(num_layers) * int(step) + int(pop_pair_idx) * int(num_layers) + int(layer_idx)
    rng = torch.Generator().manual_seed(seed)
    return tuple(torch.normal(mean=0.0, std=1.0, size=tuple(shape), generator=rng) for shape in shapes)


def vllm_dense_update_target(
    *,
    peft_name: str,
    weight_shape: tuple[int, ...],
    peft_shapes_dict: dict[str, tuple[int, ...]],
    vllm_params: dict[str, Any],
) -> tuple[Any, Any]:
    vllm_name = peft_name.replace("base_model.model.", "")

    if "self_attn.q_proj" in vllm_name:
        target_name = vllm_name.replace("self_attn.q_proj", "self_attn.qkv_proj")
        return _require_param(vllm_params, target_name, peft_name), slice(0, weight_shape[0])

    if "self_attn.k_proj" in vllm_name:
        target_name = vllm_name.replace("self_attn.k_proj", "self_attn.qkv_proj")
        q_name = peft_name.replace("k_proj", "q_proj")
        start = int(peft_shapes_dict[q_name][0])
        return _require_param(vllm_params, target_name, peft_name), slice(start, start + int(weight_shape[0]))

    if "self_attn.v_proj" in vllm_name:
        target_name = vllm_name.replace("self_attn.v_proj", "self_attn.qkv_proj")
        q_name = peft_name.replace("v_proj", "q_proj")
        k_name = peft_name.replace("v_proj", "k_proj")
        start = int(peft_shapes_dict[q_name][0]) + int(peft_shapes_dict[k_name][0])
        return _require_param(vllm_params, target_name, peft_name), slice(start, start + int(weight_shape[0]))

    if "self_attn.o_proj" in vllm_name or "mlp.down_proj" in vllm_name or "shared_expert.down_proj" in vllm_name:
        return _require_param(vllm_params, vllm_name, peft_name), (slice(None), slice(None))

    if "mlp.gate_proj" in vllm_name:
        target_name = vllm_name.replace("mlp.gate_proj", "mlp.gate_up_proj")
        return _require_param(vllm_params, target_name, peft_name), slice(0, weight_shape[0])

    if "mlp.up_proj" in vllm_name:
        target_name = vllm_name.replace("mlp.up_proj", "mlp.gate_up_proj")
        gate_name = peft_name.replace("up_proj", "gate_proj")
        start = int(peft_shapes_dict[gate_name][0])
        return _require_param(vllm_params, target_name, peft_name), slice(start, start + int(weight_shape[0]))

    if "shared_expert.gate_proj" in vllm_name:
        target_name = vllm_name.replace("shared_expert.gate_proj", "shared_expert.gate_up_proj")
        return _require_param(vllm_params, target_name, peft_name), slice(0, weight_shape[0])

    if "shared_expert.up_proj" in vllm_name:
        target_name = vllm_name.replace("shared_expert.up_proj", "shared_expert.gate_up_proj")
        gate_name = peft_name.replace("up_proj", "gate_proj")
        start = int(peft_shapes_dict[gate_name][0])
        return _require_param(vllm_params, target_name, peft_name), slice(start, start + int(weight_shape[0]))

    if "experts" in vllm_name:
        return _vllm_moe_update_target(
            vllm_name=vllm_name, peft_name=peft_name, weight_shape=weight_shape, peft_shapes_dict=peft_shapes_dict, vllm_params=vllm_params
        )

    raise RuntimeError(f"Unrecognised PEFT layer for vLLM update: {peft_name!r}")


def add_dense_update(target_param: Any, slice_obj: Any, update: Any) -> None:
    grad = update.to(dtype=target_param.dtype)
    if isinstance(slice_obj, tuple) and len(slice_obj) == 3:
        expert_idx, rows, _cols = slice_obj
        if int(expert_idx) >= target_param.shape[0]:
            return
        row_start = rows.start or 0
        row_stop = min(rows.stop or target_param.shape[1], target_param.shape[1])
        if row_start >= row_stop:
            return
        grad = grad[: row_stop - row_start]
        if grad.shape[1] > target_param.shape[2]:
            grad = grad[:, : target_param.shape[2]]
        target_param.data[int(expert_idx), row_start:row_stop, : grad.shape[1]].add_(grad)
        return

    if isinstance(slice_obj, tuple) and len(slice_obj) == 2:
        if target_param.shape[1] < grad.shape[1]:
            grad = grad[:, : target_param.shape[1]]
        target_param.data.add_(grad)
        return

    if isinstance(slice_obj, slice):
        start = int(slice_obj.start or 0)
        stop = min(int(slice_obj.stop or target_param.shape[0]), int(target_param.shape[0]))
        if start < stop:
            target_param.data[start:stop].add_(grad[: stop - start])
        return

    raise RuntimeError(f"Unsupported vLLM update slice: {slice_obj!r}")


def _vllm_moe_update_target(
    *,
    vllm_name: str,
    peft_name: str,
    weight_shape: tuple[int, ...],
    peft_shapes_dict: dict[str, tuple[int, ...]],
    vllm_params: dict[str, Any],
) -> tuple[Any, Any]:
    vllm_name_moe = vllm_name.replace(".base_layer.weight", "")
    expert_match = re.search(r"experts\.(\d+)\.", vllm_name_moe)
    if expert_match is None:
        raise RuntimeError(f"Could not parse MoE expert index from {peft_name!r}.")
    expert_idx = int(expert_match.group(1))

    if "gate_proj" in vllm_name_moe:
        target_name = re.sub(r"experts\.\d+\.gate_proj", "experts.base_layer.w13_weight", vllm_name_moe)
        return _require_param(vllm_params, target_name, peft_name), (expert_idx, slice(0, weight_shape[0]), slice(None))

    if "up_proj" in vllm_name_moe:
        target_name = re.sub(r"experts\.\d+\.up_proj", "experts.base_layer.w13_weight", vllm_name_moe)
        gate_name = peft_name.replace("up_proj", "gate_proj")
        start = int(peft_shapes_dict[gate_name][0])
        return _require_param(vllm_params, target_name, peft_name), (expert_idx, slice(start, start + int(weight_shape[0])), slice(None))

    if "down_proj" in vllm_name_moe:
        target_name = re.sub(r"experts\.\d+\.down_proj", "experts.base_layer.w2_weight", vllm_name_moe)
        return _require_param(vllm_params, target_name, peft_name), (expert_idx, slice(None), slice(None))

    raise RuntimeError(f"Unrecognised MoE PEFT layer for vLLM update: {peft_name!r}")


def _require_param(vllm_params: dict[str, Any], target_name: str, peft_name: str) -> Any:
    if target_name not in vllm_params:
        sample = list(vllm_params)[:8]
        raise RuntimeError(f"Expected vLLM parameter {target_name!r} for PEFT layer {peft_name!r}; sample keys: {sample}")
    return vllm_params[target_name]


def _jsonable_lora_config(config: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(config)
    for key, value in list(out.items()):
        if isinstance(value, set | tuple):
            out[key] = list(value)
        else:
            try:
                json.dumps(value)
            except TypeError:
                out[key] = str(value)
    return out


def _missing_runtime_message(missing: list[str]) -> str:
    return missing_runtime_message("LoRA", missing, "./ops/llm.py")


__all__ = [
    "LORA_TARGET_MODULES",
    "LoraTemplate",
    "add_dense_update",
    "build_peft_lora_template",
    "get_rng_noise",
    "materialize_lora_adapters",
    "vllm_dense_update_target",
]
