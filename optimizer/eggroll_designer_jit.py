from __future__ import annotations

from optimizer.eggroll_designer_jit_common import JittedFnConfig, apply_noiser_update
from optimizer.eggroll_rollout_helpers import rollout_step_out
from optimizer.eggroll_runtime import EggRollActionSelector


def build_jitted_fns(designer, cfg: JittedFnConfig):
    if int(getattr(cfg.env_adapter, "vector_num_envs", 1)) > 1:
        from optimizer.eggroll_designer_jit_batch import build_batched_jitted_fns

        return build_batched_jitted_fns(designer, cfg)
    jax = designer._jax
    jnp = designer._jnp
    model_cls = designer._policy.model_cls
    action_selector = EggRollActionSelector(jax, jnp, deterministic_policy=cfg.deterministic_policy)

    def rollout(params, noiser_params, thread_info, rollout_key):
        reset_key, loop_key = jax.random.split(rollout_key)
        obs, state = cfg.env_adapter.reset(reset_key)
        total_reward = jnp.array(0.0, dtype=jnp.float32)
        done = jnp.array(False)

        def step(carry, _unused):
            obs_t, state_t, total_t, done_t, key_t = carry
            key_t, action_key, env_key = jax.random.split(key_t, 3)
            policy_dist = model_cls.forward(
                cfg.noiser,
                cfg.frozen_noiser_params,
                noiser_params,
                cfg.frozen_params,
                params,
                cfg.es_tree_key,
                thread_info,
                obs_t,
            )
            action = action_selector.select_action(policy_dist, action_key)
            action = cfg.env_adapter.clip_action(action)
            next_obs, next_state, reward, terminated, truncated, _info = cfg.env_adapter.step(env_key, state_t, action)
            next_done = jnp.logical_or(terminated.astype(bool), truncated.astype(bool))
            transition = (
                obs_t,
                state_t,
                total_t,
                done_t,
                key_t,
                next_obs,
                next_state,
                reward,
                next_done,
            )
            return rollout_step_out(jax, jnp, transition)

        (_, _, total_reward, _, _), _ = jax.lax.scan(
            step,
            (obs, state, total_reward, done, loop_key),
            None,
            length=int(designer._steps_per_episode),
        )
        return total_reward

    @jax.jit
    def evaluate_population(params, noiser_params, epoch, keys):
        thread_ids = jnp.arange(keys.shape[0], dtype=jnp.int32)
        return jax.vmap(lambda thread_id, k: rollout(params, noiser_params, (epoch, thread_id), k))(thread_ids, keys)

    @jax.jit
    def evaluate_policy(params, noiser_params, keys):
        return jax.vmap(lambda k: rollout(params, noiser_params, None, k))(keys)

    @jax.jit
    def update_params(noiser_params, params, raw_scores, epoch):
        return apply_noiser_update(jnp, cfg, noiser_params, params, raw_scores, epoch)

    return evaluate_population, evaluate_policy, update_params
