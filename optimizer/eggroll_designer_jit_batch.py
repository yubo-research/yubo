from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optimizer.eggroll_designer_jit_common import JittedFnConfig, apply_noiser_update


@dataclass(frozen=True)
class _BatchedJitCtx:
    jax: Any
    jnp: Any
    model_cls: Any
    cfg: JittedFnConfig
    action_selector: Any
    env_adapter: Any
    n: int
    obs_dim: int
    steps_per_episode: int
    num_eval_envs: int


def _batched_action(
    ctx: _BatchedJitCtx,
    params,
    noiser_params,
    epoch,
    thread_id,
    obs_i,
    act_key,
):
    policy_dist = ctx.model_cls.forward(
        ctx.cfg.noiser,
        ctx.cfg.frozen_noiser_params,
        noiser_params,
        ctx.cfg.frozen_params,
        params,
        ctx.cfg.es_tree_key,
        (epoch, thread_id),
        obs_i,
    )
    return ctx.env_adapter.clip_action(ctx.action_selector.select_action(policy_dist, act_key))


def _batched_slot_action(
    slot_idx,
    obs_t,
    action_keys,
    ctx: _BatchedJitCtx,
    params,
    noiser_params,
    epoch,
):
    obs_i = ctx.jax.lax.dynamic_slice_in_dim(obs_t, slot_idx, 1, axis=0).reshape((ctx.obs_dim,))
    return _batched_action(
        ctx,
        params,
        noiser_params,
        epoch,
        slot_idx,
        obs_i,
        action_keys[slot_idx],
    )


def _batched_population_scan_step(carry, _unused, ctx: _BatchedJitCtx, params, noiser_params, epoch):
    obs_t, state_t, total_t, active_t, key_t = carry
    obs_t = ctx.jnp.reshape(obs_t, (ctx.n, ctx.obs_dim))
    keys = ctx.jax.random.split(key_t, ctx.n + 1)
    key_t = keys[0]
    action_keys = keys[1:]

    def slot_action(slot_idx):
        return _batched_slot_action(slot_idx, obs_t, action_keys, ctx, params, noiser_params, epoch)

    actions = ctx.jax.lax.map(slot_action, ctx.jnp.arange(ctx.n, dtype=ctx.jnp.int32))
    next_obs, next_state, reward, term, trunc = ctx.env_adapter.step_batched(state_t, actions)
    next_obs = ctx.jnp.reshape(next_obs, (ctx.n, ctx.obs_dim))
    done = ctx.jnp.logical_or(term.astype(ctx.jnp.bool_), trunc.astype(ctx.jnp.bool_))
    total_out = total_t + ctx.jnp.where(active_t, reward, ctx.jnp.zeros_like(reward))
    active_out = ctx.jnp.logical_and(active_t, ctx.jnp.logical_not(done))
    obs_out = ctx.jnp.where(active_out[:, None], next_obs, obs_t)
    return (obs_out, next_state, total_out, active_out, key_t), None


def _batched_eval_scan_step(carry, _unused, ctx: _BatchedJitCtx, params, noiser_params):
    obs_t, state_t, total_t, active_t, key_t = carry
    obs_t = ctx.jnp.reshape(obs_t, (ctx.n, ctx.obs_dim))
    key_t, action_key, _env_key = ctx.jax.random.split(key_t, 3)
    policy_dist = ctx.model_cls.forward(
        ctx.cfg.noiser,
        ctx.cfg.frozen_noiser_params,
        noiser_params,
        ctx.cfg.frozen_params,
        params,
        ctx.cfg.es_tree_key,
        None,
        obs_t,
    )
    actions = ctx.env_adapter.clip_action(ctx.action_selector.select_action(policy_dist, action_key))
    next_obs, next_state, reward, term, trunc = ctx.env_adapter.step_batched(state_t, actions)
    next_obs = ctx.jnp.reshape(next_obs, (ctx.n, ctx.obs_dim))
    done = ctx.jnp.logical_or(term.astype(ctx.jnp.bool_), trunc.astype(ctx.jnp.bool_))
    total_out = total_t + ctx.jnp.where(active_t, reward, ctx.jnp.zeros_like(reward))
    active_out = ctx.jnp.logical_and(active_t, ctx.jnp.logical_not(done))
    obs_out = ctx.jnp.where(active_out[:, None], next_obs, obs_t)
    return (obs_out, next_state, total_out, active_out, key_t), None


def _rollout_population(params, noiser_params, epoch, keys, ctx: _BatchedJitCtx):
    reset_key, loop_key = ctx.jax.random.split(keys[0])
    obs, state = ctx.env_adapter.reset(reset_key)
    obs = ctx.jnp.reshape(obs, (ctx.n, ctx.obs_dim))
    total = ctx.jnp.zeros((ctx.n,), dtype=ctx.jnp.float32)
    active = ctx.jnp.ones((ctx.n,), dtype=ctx.jnp.bool_)

    def step(carry, unused):
        return _batched_population_scan_step(carry, unused, ctx, params, noiser_params, epoch)

    (_, _, total, _, _), _ = ctx.jax.lax.scan(
        step,
        (obs, state, total, active, loop_key),
        None,
        length=int(ctx.steps_per_episode),
    )
    return total


def _rollout_eval(params, noiser_params, keys, ctx: _BatchedJitCtx):
    eval_n = min(int(keys.shape[0]), int(ctx.num_eval_envs), ctx.n)
    reset_key, loop_key = ctx.jax.random.split(keys[0])
    obs, state = ctx.env_adapter.reset(reset_key)
    obs = ctx.jnp.reshape(obs, (ctx.n, ctx.obs_dim))
    total = ctx.jnp.zeros((ctx.n,), dtype=ctx.jnp.float32)
    active = ctx.jnp.ones((ctx.n,), dtype=ctx.jnp.bool_)

    def step(carry, unused):
        return _batched_eval_scan_step(carry, unused, ctx, params, noiser_params)

    (_, _, total, _, _), _ = ctx.jax.lax.scan(
        step,
        (obs, state, total, active, loop_key),
        None,
        length=int(ctx.steps_per_episode),
    )
    return ctx.jnp.mean(total[:eval_n])


def build_batched_jitted_fns(designer, cfg: JittedFnConfig):
    from optimizer.eggroll_runtime import EggRollActionSelector

    env_adapter = cfg.env_adapter
    ctx = _BatchedJitCtx(
        jax=designer._jax,
        jnp=designer._jnp,
        model_cls=designer._policy.model_cls,
        cfg=cfg,
        action_selector=EggRollActionSelector(
            designer._jax,
            designer._jnp,
            deterministic_policy=cfg.deterministic_policy,
        ),
        env_adapter=env_adapter,
        n=int(env_adapter.vector_num_envs),
        obs_dim=int(env_adapter._obs_dim),
        steps_per_episode=int(designer._steps_per_episode),
        num_eval_envs=int(designer._num_envs),
    )

    @ctx.jax.jit
    def evaluate_population(params, noiser_params, epoch, keys):
        return _rollout_population(params, noiser_params, epoch, keys, ctx)

    @ctx.jax.jit
    def evaluate_policy(params, noiser_params, keys):
        return _rollout_eval(params, noiser_params, keys, ctx)

    @ctx.jax.jit
    def update_params(noiser_params, params, raw_scores, epoch):
        return apply_noiser_update(ctx.jnp, cfg, noiser_params, params, raw_scores, epoch)

    return evaluate_population, evaluate_policy, update_params
