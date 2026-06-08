from __future__ import annotations


def rollout_step_out(jax, jnp, transition):
    obs_t, state_t, total_t, done_t, key_t, next_obs, next_state, reward, next_done = transition
    active = jnp.logical_not(done_t)
    obs_out = jax.tree.map(lambda new, old: jnp.where(active, new, old), next_obs, obs_t)
    state_out = jax.tree.map(lambda new, old: jnp.where(active, new, old), next_state, state_t)
    total_out = total_t + jnp.where(active, reward, 0.0)
    done_out = jnp.logical_or(done_t, next_done)
    return (obs_out, state_out, total_out, done_out, key_t), None
