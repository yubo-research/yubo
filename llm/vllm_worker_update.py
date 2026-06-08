from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from typing import Any

from llm.lora import add_dense_update, get_rng_noise, vllm_dense_update_target


@dataclass(frozen=True)
class LoraUpdateContext:
    torch: Any
    device: Any
    normalized: list[float]
    peft_shapes_dict: dict[str, tuple[int, ...]]
    vllm_params: dict[str, Any]
    pop_step: int
    chunk_size: int
    args: Any

    @property
    def num_pop_pairs(self) -> int:
        return int(self.args.population_size) // 2


def apply_lora_es_update(
    worker: Any,
    normalized_fitnesses: list[float],
    peft_shapes_dict: dict[str, tuple[int, ...]],
    es_step: int,
    args: Any,
) -> bool:
    import torch

    if getattr(worker, "gpu_rank", 0) != 0:
        return False
    ctx = LoraUpdateContext(
        torch=torch,
        device=worker.device,
        normalized=[float(x) for x in normalized_fitnesses],
        peft_shapes_dict=peft_shapes_dict,
        vllm_params=dict(worker.model_runner.model.named_parameters()),
        pop_step=int(es_step) // int(args.steps_per_adapter),
        chunk_size=update_chunk_size(int(args.population_size), int(args.lora_r)),
        args=args,
    )
    for layer_idx, item in enumerate(peft_shapes_dict.items()):
        _apply_layer_update(ctx, layer_idx, item)
    _sync_cuda(torch)
    gc.collect()
    return True


def update_chunk_size(population_size: int, lora_rank: int) -> int:
    if lora_rank <= 2:
        return min(128, population_size // 2)
    if lora_rank <= 8:
        return min(64, population_size // 2)
    return min(32, population_size // 2)


def _apply_layer_update(ctx: LoraUpdateContext, layer_idx: int, item: tuple[str, tuple[int, ...]]) -> None:
    peft_name, raw_shape = item
    weight_shape = tuple(int(dim) for dim in raw_shape)
    layer_update = ctx.torch.zeros(weight_shape, device=ctx.device, dtype=ctx.torch.float32)
    lora_a_shape = (int(ctx.args.lora_r), weight_shape[1])
    lora_b_shape = (weight_shape[0], int(ctx.args.lora_r))
    for chunk_start in range(0, ctx.num_pop_pairs, ctx.chunk_size):
        chunk = _weighted_chunk_update(ctx, layer_idx, lora_a_shape, lora_b_shape, chunk_start)
        layer_update.add_(chunk)
        del chunk
    _commit_layer_update(ctx, peft_name, weight_shape, layer_update)


def _weighted_chunk_update(
    ctx: LoraUpdateContext,
    layer_idx: int,
    lora_a_shape: tuple[int, ...],
    lora_b_shape: tuple[int, ...],
    chunk_start: int,
):
    chunk_end = min(chunk_start + ctx.chunk_size, ctx.num_pop_pairs)
    noise_a_batch, noise_b_batch, fitness_diffs = _chunk_noise_batches(
        ctx,
        layer_idx=layer_idx,
        lora_a_shape=lora_a_shape,
        lora_b_shape=lora_b_shape,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
    )
    weighted = _weighted_noise(ctx.torch, noise_a_batch, noise_b_batch, fitness_diffs, int(ctx.args.lora_r))
    del noise_a_batch, noise_b_batch, fitness_diffs
    return weighted


def _chunk_noise_batches(
    ctx: LoraUpdateContext,
    *,
    layer_idx: int,
    lora_a_shape: tuple[int, ...],
    lora_b_shape: tuple[int, ...],
    chunk_start: int,
    chunk_end: int,
):
    noise_a_list = []
    noise_b_list = []
    fitness_diffs = []
    for pop_pair_idx in range(chunk_start, chunk_end):
        fitness_diffs.append(ctx.normalized[pop_pair_idx * 2] - ctx.normalized[pop_pair_idx * 2 + 1])
        noise_a, noise_b = get_rng_noise(
            base_seed=int(ctx.args.base_seed),
            num_pop_pairs=ctx.num_pop_pairs,
            pop_pair_idx=pop_pair_idx,
            num_layers=len(ctx.peft_shapes_dict),
            layer_idx=layer_idx,
            step=ctx.pop_step,
            shapes=[lora_a_shape, lora_b_shape],
        )
        noise_a_list.append(noise_a)
        noise_b_list.append(noise_b)
    noise_a_batch = ctx.torch.stack(noise_a_list).to(ctx.device) * math.sqrt(float(ctx.args.sigma))
    noise_b_batch = ctx.torch.stack(noise_b_list).to(ctx.device) * math.sqrt(float(ctx.args.sigma) / float(ctx.args.lora_r))
    diffs = ctx.torch.tensor(fitness_diffs, device=ctx.device, dtype=noise_a_batch.dtype)
    return noise_a_batch, noise_b_batch, diffs


def _weighted_noise(torch, noise_a_batch, noise_b_batch, fitness_diffs, lora_rank: int):
    if int(lora_rank) == 1:
        weighted_b = noise_b_batch.squeeze(2) * fitness_diffs.unsqueeze(1)
        return torch.mm(weighted_b.t(), noise_a_batch.squeeze(1))
    noise_batch = torch.bmm(noise_b_batch, noise_a_batch)
    weighted = (noise_batch * fitness_diffs.view(-1, 1, 1)).sum(dim=0)
    del noise_batch
    return weighted


def _commit_layer_update(ctx: LoraUpdateContext, peft_name: str, weight_shape: tuple[int, ...], layer_update) -> None:
    scale = float(ctx.args.learning_rate) / (int(ctx.args.population_size) * float(ctx.args.sigma) + 1e-8)
    gradient = layer_update * scale
    if bool(ctx.args.scale_lr_in_grad):
        gradient *= math.sqrt(int(ctx.args.population_size))
    target_param, slice_obj = vllm_dense_update_target(
        peft_name=peft_name,
        weight_shape=weight_shape,
        peft_shapes_dict=ctx.peft_shapes_dict,
        vllm_params=ctx.vllm_params,
    )
    add_dense_update(target_param, slice_obj, gradient)
    _empty_cuda_cache(ctx.torch)


def _empty_cuda_cache(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync_cuda(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _apply_subspace_via_groups(groups, vllm_params, noise, scale) -> None:
    for name, g_info in groups.items():
        param = vllm_params.get(name)
        if param is None:
            continue
        updates = noise[g_info["subspace_indices"]]
        updates.mul_(g_info["signs"])
        if scale != 1.0:
            updates.mul_(scale)
        param.data.view(-1).index_add_(0, g_info["param_indices"], updates)


def _apply_subspace_fallback(template, vllm_params, noise, search_dim) -> None:
    for i in range(search_dim):
        delta = noise[i].item() * template.basis_sign[i]
        if abs(delta) < 1e-15:
            continue
        param_meta = template.parameters[template.basis_leaf[i]]
        param = vllm_params.get(param_meta.name)
        if param is None:
            continue
        flat_idx = template.basis_index[i]
        param.data.view(-1)[flat_idx].add_(delta)


def apply_subspace_perturbation(
    worker: Any,
    template: Any | None,
    seed: int,
    scale: float,
) -> bool:
    """Applies a random coordinate-based perturbation to model parameters."""
    import torch

    if template is None:
        template = getattr(worker, "_universal_subspace_template", None)
    if template is None:
        raise RuntimeError("Universal subspace template not set on vLLM worker.")

    device = worker.device
    search_dim = int(template.search_dim)

    # 1. Sample subspace noise
    g = torch.Generator(device=str(device))
    g.manual_seed(seed)
    noise = torch.randn((search_dim,), device=device, generator=g)
    noise.mul_(scale)

    # 2. Apply to coordinates
    vllm_params = getattr(worker, "_universal_named_parameters", None)
    if not isinstance(vllm_params, dict):
        vllm_params = dict(worker.model_runner.model.named_parameters())

    groups = getattr(worker, "_universal_update_groups", None)
    if groups is not None:
        _apply_subspace_via_groups(groups, vllm_params, noise, scale)
        return True

    _apply_subspace_fallback(template, vllm_params, noise, search_dim)
    return True


def _accumulate_universal_es_grad(z_grad, normalized_fitnesses, num_pop_pairs, args, template, device, torch, es_step):
    from llm.universal_subspace import universal_subspace_seed

    search_dim = int(template.search_dim)
    pop_step = int(es_step) // int(args.steps_per_adapter)
    for pop_pair_idx in range(num_pop_pairs):
        diff = normalized_fitnesses[pop_pair_idx * 2] - normalized_fitnesses[pop_pair_idx * 2 + 1]
        if abs(diff) < 1e-8:
            continue
        seed = universal_subspace_seed(
            base_seed=int(args.base_seed),
            num_pop_pairs=num_pop_pairs,
            search_dim=search_dim,
            pop_step=pop_step,
            pop_pair_idx=pop_pair_idx,
        )
        g = torch.Generator(device=str(device))
        g.manual_seed(seed)
        noise = torch.randn((search_dim,), device=device, generator=g)
        z_grad.add_(noise, alpha=diff)


def apply_universal_es_update(
    worker: Any,
    normalized_fitnesses: list[float],
    template: Any | None,
    es_step: int,
    args: Any,
) -> bool:
    """Applies a global coordinate-based subspace update to all model parameters."""
    import torch

    if getattr(worker, "gpu_rank", 0) != 0:
        return False

    if template is None:
        template = getattr(worker, "_universal_subspace_template", None)
    if template is None:
        raise RuntimeError("Universal subspace template not set on vLLM worker.")

    device = worker.device
    num_pop_pairs = int(args.population_size) // 2

    # 1. Reconstruct the update in subspace (size search_dim)
    search_dim = int(template.search_dim)
    z_grad = torch.zeros((search_dim,), device=device, dtype=torch.float32)
    _accumulate_universal_es_grad(z_grad, normalized_fitnesses, num_pop_pairs, args, template, device, torch, es_step)

    # 2. Scale gradient
    scale = float(args.learning_rate) / (int(args.population_size) * float(args.sigma) + 1e-8)
    if bool(args.scale_lr_in_grad):
        scale *= math.sqrt(int(args.population_size))
    z_grad.mul_(scale)

    # 3. Project subspace gradient to model parameters
    vllm_params = getattr(worker, "_universal_named_parameters", None)
    if not isinstance(vllm_params, dict):
        vllm_params = dict(worker.model_runner.model.named_parameters())

    groups = getattr(worker, "_universal_update_groups", None)
    if groups is not None:
        for name, g_info in groups.items():
            param = vllm_params.get(name)
            if param is None:
                continue

            # Vectorized projection: maps subspace grad back to chooses indices
            updates = z_grad[g_info["subspace_indices"]]
            updates.mul_(g_info["signs"])
            param.data.view(-1).index_add_(0, g_info["param_indices"], updates)
    else:
        _apply_universal_es_update_fallback(z_grad, template, vllm_params)

    _sync_cuda(torch)
    gc.collect()
    return True


def _apply_universal_es_update_fallback(z_grad, template, vllm_params) -> None:
    search_dim = int(template.search_dim)
    for i in range(search_dim):
        grad_val = z_grad[i].item() * template.basis_sign[i]
        if abs(grad_val) < 1e-15:
            continue

        param_meta = template.parameters[template.basis_leaf[i]]
        param = vllm_params.get(param_meta.name)
        if param is None:
            continue

        # Apply coordinate update
        flat_idx = template.basis_index[i]
        param.data.view(-1)[flat_idx].add_(grad_val)
