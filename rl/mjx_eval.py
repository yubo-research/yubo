from __future__ import annotations

from problems.jax_env_core import _normalize_obs


def make_mjx_eval_step(config, runtime, action_fn):
    jax, jnp, adapter = runtime.jax, runtime.jnp, runtime.adapter
    num_envs = int(config.eval.num_envs)
    num_steps = int(config.eval.num_steps)

    def evaluate(params, rms, key):
        key, reset_key, rollout_key = jax.random.split(key, 3)
        reset_keys = jax.random.split(reset_key, num_envs)
        obs, env_state = jax.vmap(adapter.reset)(reset_keys)

        def step(carry, _):
            obs_t, env_state_t, key_t, ret_t, len_t, done_t = carry
            key_t, _act_key, env_key = jax.random.split(key_t, 3)

            norm_obs = _normalize_obs(obs_t, rms, jnp)
            action = action_fn(params, jnp, norm_obs, runtime)
            step_out = jax.vmap(adapter.step)(jax.random.split(env_key, num_envs), env_state_t, action)
            done = jnp.logical_or(step_out.terminated.astype(bool), step_out.truncated.astype(bool)).astype(jnp.float32)
            reward = step_out.reward * (1.0 - done_t)
            new_ret = ret_t + reward
            new_len = len_t + (1.0 - done_t)
            new_done = jnp.logical_or(done_t.astype(bool), done.astype(bool)).astype(jnp.float32)
            return (
                step_out.obs,
                step_out.state,
                key_t,
                new_ret,
                new_len,
                new_done,
            ), None

        init_ret = jnp.zeros((num_envs,), dtype=jnp.float32)
        init_len = jnp.zeros((num_envs,), dtype=jnp.float32)
        init_done = jnp.zeros((num_envs,), dtype=jnp.float32)
        (_obs, _state, _key, final_ret, _final_len, _final_done), _ = jax.lax.scan(
            step,
            (obs, env_state, rollout_key, init_ret, init_len, init_done),
            None,
            length=num_steps,
        )
        return jnp.mean(final_ret)

    return jax.jit(evaluate)
