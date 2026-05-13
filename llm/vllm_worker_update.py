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
