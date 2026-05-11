from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from problems.pre_obj_stack import _HyperscaleESStack, _log


@dataclass(frozen=True)
class _GenerateThreadContext:
    stack: _HyperscaleESStack
    model: Any
    noiser: Any
    frozen_noiser_params: Any
    config: Any
    base_evo_keys: Any
    master_gen_key: Any
    temperature: float


@dataclass(frozen=True)
class _ValidateContext:
    stack: _HyperscaleESStack
    model: Any
    config: Any
    params_example: Any
    base_evo_keys: Any
    master_gen_key: Any
    tokenizer: Any
    legacy_tokenizer: Any
    args: Any


def _fold_in_thread_key(jax, key, epoch, thread_idx):
    return jax.random.fold_in(jax.random.fold_in(key, epoch), thread_idx)


def _build_generate_thread(ctx: _GenerateThreadContext):
    jax = ctx.stack.jax
    jnp = ctx.stack.jnp

    def forward_and_sample(noiser_params, params, input_token, input_state, generation_key, iterinfo):
        gen_key, sample_key = jax.random.split(generation_key)
        logits, generated_state = ctx.model.forward(
            ctx.noiser,
            ctx.frozen_noiser_params,
            noiser_params,
            ctx.config,
            params,
            ctx.base_evo_keys,
            iterinfo,
            input_token,
            input_state,
        )
        if float(ctx.temperature) != 0.0:
            sampled_token = jax.random.categorical(sample_key, logits[-1] / float(ctx.temperature))
        else:
            sampled_token = jnp.argmax(logits[-1])
        return sampled_token, generated_state, gen_key

    def generate_thread(noiser_params, params, prompt, thread_idx, epoch):
        gen_key = _fold_in_thread_key(jax, ctx.master_gen_key, epoch, thread_idx)
        iterinfo = (epoch, thread_idx)

        def step(carry, input_token):
            token, state, key = carry
            true_input = jnp.where(input_token == 0, token, input_token)
            next_token, next_state, next_key = forward_and_sample(noiser_params, params, true_input, state, key, iterinfo)
            return (next_token, next_state, next_key), true_input

        init_token = jnp.asarray(0, dtype=jnp.int32)
        init_state = ctx.model.default_state(params, ctx.config)
        _, output_tokens = jax.lax.scan(step, (init_token, init_state, gen_key), prompt)
        return output_tokens

    return generate_thread


def _build_validate(
    ctx: _ValidateContext,
    *,
    temperature: float = 0.0,
    use_validation_set: bool = True,
    noiser_cls,
    sigma: float = 0.0,
):
    jax = ctx.stack.jax
    jnp = ctx.stack.jnp
    frozen_noiser_params, noiser_params = noiser_cls.init_noiser(ctx.params_example, float(sigma), 0.0)
    task_registry = ctx.stack.validation_tasks if use_validation_set else ctx.stack.all_tasks
    task = task_registry[ctx.args.task](ctx.tokenizer, ctx.legacy_tokenizer, int(ctx.args.generation_length))
    generate_thread = _build_generate_thread(
        _GenerateThreadContext(
            stack=ctx.stack,
            model=ctx.model,
            noiser=noiser_cls,
            frozen_noiser_params=frozen_noiser_params,
            config=ctx.config,
            base_evo_keys=ctx.base_evo_keys,
            master_gen_key=ctx.master_gen_key,
            temperature=float(temperature),
        )
    )

    parallel_validations = int(ctx.args.parallel_validations)
    generation_length = int(ctx.args.generation_length)
    _log("compiling validation generator")
    t0 = time.perf_counter()
    generate_batch = (
        jax.jit(jax.vmap(generate_thread, in_axes=(None, None, 0, 0, None)))
        .lower(
            noiser_params,
            ctx.params_example,
            jax.ShapeDtypeStruct((parallel_validations, generation_length), jnp.dtype("int32")),
            jnp.arange(parallel_validations),
            0,
        )
        .compile()
    )
    _log(f"validation generator ready dt={time.perf_counter() - t0:.2f}s")

    def validate(params, epoch: int):
        total_score = 0.0
        cpu = jax.local_devices(backend="cpu")[0]
        for i in range(int(ctx.args.validation_iterations)):
            indices = jnp.arange(parallel_validations) + (i * parallel_validations)
            prompts = task.get_input(indices)
            output_batch = jax.block_until_ready(generate_batch(noiser_params, params, prompts, indices, int(epoch)))
            scores = task.get_batch_fitness(jax.device_put(indices, cpu), jax.device_put(output_batch, cpu))
            total_score += jnp.sum(jnp.asarray(scores))
        return total_score / (parallel_validations * int(ctx.args.validation_iterations))

    return validate
