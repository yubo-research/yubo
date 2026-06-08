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


def _build_validate_coeff_batch(validate_ctx: _ValidateContext, codec, *, noiser_cls, sigma: float = 0.0):
    jax = validate_ctx.stack.jax
    jnp = validate_ctx.stack.jnp
    frozen_noiser_params, noiser_params = noiser_cls.init_noiser(validate_ctx.params_example, float(sigma), 0.0)
    task = validate_ctx.stack.validation_tasks[validate_ctx.args.task](
        validate_ctx.tokenizer,
        validate_ctx.legacy_tokenizer,
        int(validate_ctx.args.generation_length),
    )
    generate_thread = _build_generate_thread(
        _GenerateThreadContext(
            stack=validate_ctx.stack,
            model=validate_ctx.model,
            noiser=noiser_cls,
            frozen_noiser_params=frozen_noiser_params,
            config=validate_ctx.config,
            base_evo_keys=validate_ctx.base_evo_keys,
            master_gen_key=validate_ctx.master_gen_key,
            temperature=0.0,
        )
    )
    generate_prompt_batch = jax.vmap(generate_thread, in_axes=(None, None, 0, 0, None))
    parallel_validations = int(validate_ctx.args.parallel_validations)

    @jax.jit
    def generate_coeff_batch(params, coeff_batch, prompts, indices, epoch):
        def generate_coeff(coeffs):
            candidate_params = codec.decode_device(coeffs, params=params)
            return generate_prompt_batch(noiser_params, candidate_params, prompts, indices, epoch)

        # Keep one decoded 2.9B tree live at a time while removing Python dispatch
        # and host-side subspace decoding from the candidate loop.
        return jax.lax.map(generate_coeff, coeff_batch)

    def validate_many(coeff_batch, epoch: int):
        coeff_batch = jnp.asarray(coeff_batch, dtype=jnp.float32)
        if coeff_batch.ndim != 2 or coeff_batch.shape[1] != codec.dim:
            raise ValueError(f"coeff_batch must have shape (n, {codec.dim}), got {coeff_batch.shape}.")
        total_scores = jnp.zeros((coeff_batch.shape[0],), dtype=jnp.float32)
        cpu = jax.local_devices(backend="cpu")[0]
        for i in range(int(validate_ctx.args.validation_iterations)):
            indices = jnp.arange(parallel_validations) + (i * parallel_validations)
            prompts = task.get_input(indices)
            output_batch = generate_coeff_batch(validate_ctx.params_example, coeff_batch, prompts, indices, int(epoch))
            output_batch = jax.device_put(jax.block_until_ready(output_batch), cpu)
            score_rows = [task.get_batch_fitness(jax.device_put(indices, cpu), candidate_outputs) for candidate_outputs in output_batch]
            total_scores += jnp.asarray([jnp.sum(row) for row in score_rows])
        return total_scores / (parallel_validations * int(validate_ctx.args.validation_iterations))

    return validate_many
