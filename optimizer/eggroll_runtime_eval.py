from __future__ import annotations

import numpy as np

from optimizer.eggroll_rollout_helpers import rollout_step_out


class EggRollRuntimeEvaluator:
    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self._stateful_eval_key = runtime.eval_key_base
        self._evaluate_batch_jit = self._build()

    def _build(self):
        rt = self._runtime
        jax = rt.jax
        jnp = rt.jnp
        env_adapter = rt.env_adapter
        model_cls = rt.policy.model_cls
        noiser = rt.identity_noiser
        frozen_params = rt.policy.frozen_params
        es_tree_key = rt.es_tree_key
        steps_per_episode = int(rt.steps_per_episode)

        def rollout_one(params, rollout_key):
            reset_key, loop_key = jax.random.split(rollout_key)
            obs, state = env_adapter.reset(reset_key)
            total_reward = jnp.array(0.0, dtype=jnp.float32)
            done = jnp.array(False)

            def step(carry, _unused):
                obs_t, state_t, total_t, done_t, key_t = carry
                key_t, action_key, env_key = jax.random.split(key_t, 3)
                policy_dist = model_cls.forward(noiser, None, None, frozen_params, params, es_tree_key, None, obs_t)
                action = rt.action_selector.select_action(policy_dist, action_key)
                next_obs, next_state, reward, terminated, truncated, _info = env_adapter.step(env_key, state_t, env_adapter.clip_action(action))
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
                length=steps_per_episode,
            )
            return total_reward

        def evaluate_candidate(x, keys):
            params = rt.decode_vector_params(x)
            scores = jax.vmap(lambda k: rollout_one(params, k))(keys)
            mean = jnp.mean(scores)
            se = jnp.where(
                scores.shape[0] > 1,
                jnp.std(scores) / jnp.sqrt(scores.shape[0]),
                jnp.array(0.0, dtype=jnp.float32),
            )
            return mean, se

        @jax.jit
        def evaluate_batch(x_batch, keys_batch):
            means, ses = jax.vmap(evaluate_candidate)(x_batch, keys_batch)
            return means, ses

        return evaluate_batch

    def keys_for_seed(self, seed: int, num_candidates: int):
        rt = self._runtime
        keys = rt.jax.random.split(
            rt.jax.random.fold_in(rt.eval_key_base, int(seed) & 0xFFFFFFFF),
            int(num_candidates) * rt.num_envs,
        )
        return keys.reshape((int(num_candidates), rt.num_envs))

    def next_eval_keys(self, num_candidates: int):
        self._stateful_eval_key, batch_key = self._runtime.jax.random.split(self._stateful_eval_key)
        keys = self._runtime.jax.random.split(batch_key, int(num_candidates) * self._runtime.num_envs)
        return keys.reshape((int(num_candidates), self._runtime.num_envs))

    def evaluate(self, x: np.ndarray, *, seed: int) -> tuple[float, float]:
        means, ses = self.evaluate_many(self._runtime.stack_vectors((x,)), seed=int(seed))
        return float(means[0]), float(ses[0])

    def evaluate_many(self, x_batch: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        x_batch = self._runtime.to_vector_batch(x_batch)
        return self.evaluate_many_with_keys(x_batch, self.keys_for_seed(int(seed), int(x_batch.shape[0])))

    def evaluate_many_with_keys(self, x_batch: np.ndarray, keys_batch) -> tuple[np.ndarray, np.ndarray]:
        rt = self._runtime
        x_batch = rt.to_vector_batch(x_batch)
        if x_batch.ndim != 2 or x_batch.shape[1] != rt.dim:
            raise ValueError(f"x_batch must have shape (n, {rt.dim}), got {x_batch.shape}.")
        means, ses = self._evaluate_batch_jit(x_batch, keys_batch)
        means = np.asarray(rt.jax.block_until_ready(means), dtype=np.float64)
        ses = np.asarray(rt.jax.block_until_ready(ses), dtype=np.float64)
        return means, ses

    def evaluate_values_with_keys(self, x_batch: np.ndarray, keys_batch) -> np.ndarray:
        means, _ses = self.evaluate_many_with_keys(x_batch, keys_batch)
        return means
