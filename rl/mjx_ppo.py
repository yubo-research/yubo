from __future__ import annotations

import numpy as np

from problems.jax_env_core import (
    _init_reward_rms,
    _init_rms,
    _normalize_obs,
    _normalize_reward,
    _update_reward_rms,
    _update_rms,
)
from rl import registry
from rl.mjx_eval import make_mjx_eval_step
from rl.mjx_ppo_config import MJXPPOConfig
from rl.mjx_ppo_loop import (
    MJXPPOResult,
    _AgentState,
    _checkpoint_fn,
    _eval_args,
    _result,
    _TrainState,
)
from rl.mjx_runtime import (
    MJXRuntime as _Runtime,
)
from rl.mjx_runtime import (
    make_mjx_runtime as _make_runtime,
)
from rl.mjx_train_loop import run_mjx_training_loop


def _init_layer(jax, jnp, key, in_dim: int, out_dim: int):
    w_key, _b_key = jax.random.split(key)
    scale = np.sqrt(2.0 / float(in_dim))
    return {
        "w": jax.random.normal(w_key, (in_dim, out_dim)) * scale,
        "b": jnp.zeros((out_dim,)),
    }


def _init_params(jax, jnp, key, obs_dim: int, act_dim: int, hidden: int):
    keys = jax.random.split(key, 5)
    return {
        "policy_1": _init_layer(jax, jnp, keys[0], obs_dim, hidden),
        "policy_2": _init_layer(jax, jnp, keys[1], hidden, hidden),
        "policy_mean": _init_layer(jax, jnp, keys[2], hidden, act_dim),
        "value_1": _init_layer(jax, jnp, keys[3], obs_dim, hidden),
        "value_2": _init_layer(jax, jnp, keys[4], hidden, hidden),
        "value_out": {"w": jnp.zeros((hidden, 1)), "b": jnp.zeros((1,))},
        "log_std": jnp.full((act_dim,), -0.5),
    }


def _linear(params, x):
    return x @ params["w"] + params["b"]


def _policy(params, jnp, obs):
    h = jnp.tanh(_linear(params["policy_1"], obs))
    h = jnp.tanh(_linear(params["policy_2"], h))
    return _linear(params["policy_mean"], h), jnp.exp(params["log_std"])


def _value(params, jnp, obs):
    h = jnp.tanh(_linear(params["value_1"], obs))
    h = jnp.tanh(_linear(params["value_2"], h))
    return jnp.squeeze(_linear(params["value_out"], h), axis=-1)


def _normal_log_prob(jnp, action, mean, std):
    z = (action - mean) / std
    log_prob = -0.5 * z * z - jnp.log(std) - 0.5 * jnp.log(2.0 * jnp.pi)
    return jnp.sum(log_prob, axis=-1)


def _eval_action(params, jnp, obs, runtime):
    mean, _std = _policy(params, jnp, obs)
    return jnp.clip(mean, runtime.low, runtime.high)


def _sample_action(jax, jnp, params, obs, key, low, high):
    mean, std = _policy(params, jnp, obs)
    action = mean + std * jax.random.normal(key, mean.shape)
    env_action = jnp.clip(action, low, high)
    log_prob = _normal_log_prob(jnp, action, mean, std)
    return action, env_action, log_prob


def _init_train_state(config: MJXPPOConfig, runtime: _Runtime):
    jax, jnp, optax = runtime.jax, runtime.jnp, runtime.optax
    key = jax.random.key(int(config.seed))
    key, actor_key, reset_key = jax.random.split(key, 3)
    params = _init_params(jax, jnp, actor_key, runtime.obs_dim, runtime.act_dim, int(config.hidden_size))
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(config.optim.max_grad_norm)),
        optax.scale_by_adam(),
    )
    opt_state = optimizer.init(params)
    num_envs = int(config.collector.num_envs)
    reset_keys = jax.random.split(reset_key, num_envs)
    obs, env_state = jax.vmap(runtime.adapter.reset)(reset_keys)

    # Initialize RMS
    rms = _init_rms(jnp, (runtime.obs_dim,))
    reward_rms = _init_reward_rms(jnp)
    discounted_return = jnp.zeros((num_envs,), dtype=jnp.float32)

    state = _TrainState(
        iteration=jnp.asarray(0, dtype=jnp.int32),
        params=params,
        opt_state=opt_state,
        obs_rms=rms,
        reward_rms=reward_rms,
        discounted_return=discounted_return,
        obs=obs,
        env_state=env_state,
        running_return=jnp.zeros((num_envs,), dtype=jnp.float32),
        running_length=jnp.zeros((num_envs,), dtype=jnp.int32),
        key=key,
    )
    return state, optimizer


def _ppo_learning_rate(jnp, config: MJXPPOConfig, iteration, num_iterations: int):
    if not bool(config.optim.anneal_lr):
        return jnp.asarray(float(config.optim.lr), dtype=jnp.float32)
    iteration_f = jnp.asarray(iteration, dtype=jnp.float32)
    frac = 1.0 - (iteration_f - 1.0) / float(num_iterations)
    return jnp.asarray(float(config.optim.lr), dtype=jnp.float32) * frac


def _ppo_loss(jnp, config: MJXPPOConfig, params, rms, batch):
    norm_obs = _normalize_obs(batch["obs"], rms, jnp)
    mean, std = _policy(params, jnp, norm_obs)
    log_prob = _normal_log_prob(jnp, batch["action"], mean, std)
    value = _value(params, jnp, norm_obs)

    logratio = log_prob - batch["log_prob"]
    ratio = jnp.exp(logratio)
    surr1 = ratio * batch["advantage"]
    surr2 = (
        jnp.clip(
            ratio,
            1.0 - float(config.loss.clip_epsilon),
            1.0 + float(config.loss.clip_epsilon),
        )
        * batch["advantage"]
    )
    loss_pi = -jnp.mean(jnp.minimum(surr1, surr2))

    if bool(config.loss.clip_value_loss):
        value_clipped = batch["value"] + jnp.clip(
            value - batch["value"],
            -float(config.loss.clip_epsilon),
            float(config.loss.clip_epsilon),
        )
        v_loss1 = jnp.square(value - batch["target"])
        v_loss2 = jnp.square(value_clipped - batch["target"])
        loss_v = jnp.mean(jnp.maximum(v_loss1, v_loss2))
    else:
        loss_v = jnp.mean(jnp.square(value - batch["target"]))

    entropy = jnp.mean(jnp.sum(jnp.log(std) + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1))
    loss_ent = -float(config.loss.entropy_coeff) * entropy
    total_loss = loss_pi + float(config.loss.critic_coeff) * loss_v + loss_ent
    return total_loss, (loss_pi, loss_v, entropy, ratio, logratio)


def _make_train_step(config: MJXPPOConfig, runtime: _Runtime, optimizer):
    jax, jnp = runtime.jax, runtime.jnp
    optax = runtime.optax
    adapter = runtime.adapter
    num_envs = int(config.collector.num_envs)
    num_steps = int(config.collector.num_steps)
    minibatch_size = int(config.optim.minibatch_size)
    num_epochs = int(config.optim.num_epochs)
    frames_per_iter = num_envs * num_steps
    num_iterations = max(1, int(config.collector.total_frames) // frames_per_iter)

    def _train_step(state: _TrainState):
        iteration = jnp.asarray(state.iteration, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
        lr_now = _ppo_learning_rate(jnp, config, iteration, num_iterations)

        def rollout_step(
            carry,
            _,
        ):
            (
                obs_t,
                state_t,
                key_t,
                rms_t,
                r_rms_t,
                disc_return_t,
                running_return_t,
                running_length_t,
            ) = carry
            key_t, act_key, env_key = jax.random.split(key_t, 3)

            # Use the RMS from the start of the rollout for consistency
            norm_obs = _normalize_obs(obs_t, rms_t, jnp)
            action, env_action, log_prob = _sample_action(jax, jnp, state.params, norm_obs, act_key, runtime.low, runtime.high)

            step_out = jax.vmap(adapter.step)(jax.random.split(env_key, num_envs), state_t, env_action)
            next_obs = step_out.obs
            next_state = step_out.state
            reward = step_out.reward
            terminated = step_out.terminated
            truncated = step_out.truncated

            # Update running stats for the NEXT rollout
            rms_t = _update_rms(rms_t, next_obs, jnp)
            done = jnp.logical_or(terminated.astype(bool), truncated.astype(bool)).astype(jnp.float32)
            disc_return_t = reward + config.loss.gamma * disc_return_t
            r_rms_t = _update_reward_rms(r_rms_t, disc_return_t, jnp)
            disc_return_t = disc_return_t * (1.0 - done)

            # Episodic stats
            new_running_return = running_return_t + reward
            new_running_length = running_length_t + 1

            # Record stats of finished episodes
            ep_return = new_running_return * done
            ep_length = new_running_length.astype(jnp.float32) * done

            # Reset finished envs
            next_running_return = jnp.where(done.astype(bool), 0.0, new_running_return)
            next_running_length = jnp.where(done.astype(bool), 0, new_running_length)

            # Use the reward RMS from the start of the rollout for consistency
            train_reward = reward
            if bool(config.loss.normalize_reward):
                train_reward = _normalize_reward(reward, state.reward_rms, jnp)

            transition = {
                "obs": obs_t,
                "action": action,
                "log_prob": log_prob,
                "reward": train_reward,
                "raw_reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "ep_return": ep_return,
                "ep_length": ep_length,
                "value": _value(state.params, jnp, norm_obs),
            }
            return (
                next_obs,
                next_state,
                key_t,
                rms_t,
                r_rms_t,
                disc_return_t,
                next_running_return,
                next_running_length,
            ), transition

        initial_carry = (
            state.obs,
            state.env_state,
            state.key,
            state.obs_rms,
            state.reward_rms,
            state.discounted_return,
            state.running_return,
            state.running_length,
        )
        (
            (
                next_obs,
                next_env_state,
                next_key,
                next_rms,
                next_reward_rms,
                next_discounted_return,
                next_running_return,
                next_running_length,
            ),
            data,
        ) = jax.lax.scan(rollout_step, initial_carry, None, length=num_steps)
        rollout_raw_reward = data["raw_reward"]

        # Batch-ify data: [steps, num_envs, ...] -> [steps * num_envs, ...]
        def flatten(x):
            return x.reshape((-1,) + x.shape[2:])

        data = jax.tree_util.tree_map(flatten, data)

        # GAE
        # Data is currently flattened: [num_steps * num_envs, ...]
        # We need it structured for GAE: [num_steps, num_envs, ...]
        def unflatten(x):
            return x.reshape((num_steps, num_envs) + x.shape[1:])

        gae_data = jax.tree_util.tree_map(unflatten, data)

        last_norm_obs = _normalize_obs(next_obs, next_rms, jnp)
        next_value = _value(state.params, jnp, last_norm_obs)

        def compute_gae(carry, transition):
            next_v, next_gae = carry
            done = jnp.logical_or(
                transition["terminated"].astype(bool),
                transition["truncated"].astype(bool),
            ).astype(jnp.float32)
            mask = 1.0 - done
            delta = transition["reward"] + config.loss.gamma * next_v * mask - transition["value"]
            gae = delta + config.loss.gamma * config.loss.gae_lambda * mask * next_gae
            return (transition["value"], gae), gae

        _, advantages = jax.lax.scan(
            compute_gae,
            (next_value, jnp.zeros_like(next_value)),
            gae_data,
            reverse=True,
        )
        # Flatten back for training
        data["advantage"] = flatten(advantages)
        data["target"] = data["advantage"] + data["value"]

        if bool(config.loss.normalize_advantage):
            data["advantage"] = (data["advantage"] - jnp.mean(data["advantage"])) / (jnp.std(data["advantage"]) + 1e-8)

        def update_minibatch(carry, batch):
            params, opt_state, rms = carry
            grad_fn = jax.value_and_grad(lambda p: _ppo_loss(jnp, config, p, rms, batch), has_aux=True)
            (loss, (l_pi, l_v, ent, rat, lograt)), grads = grad_fn(params)
            updates, next_opt_state = optimizer.update(grads, opt_state, params)
            updates = jax.tree_util.tree_map(lambda value: -lr_now * value, updates)
            next_params = optax.apply_updates(params, updates)

            kl = jnp.mean((rat - 1.0) - lograt)
            clipfrac = jnp.mean(jnp.abs(rat - 1.0) > float(config.loss.clip_epsilon)).astype(jnp.float32)

            return (next_params, next_opt_state, rms), (
                loss,
                l_pi,
                l_v,
                ent,
                kl,
                clipfrac,
            )

        def update_epoch(carry, _):
            params, opt_state, rms, ek = carry
            ek, subkey = jax.random.split(ek)
            permutation = jax.random.permutation(subkey, num_envs * num_steps)
            ordered = jax.tree_util.tree_map(lambda x: x[permutation], data)
            usable = (num_envs * num_steps // minibatch_size) * minibatch_size
            batches = {k: value[:usable].reshape((-1, minibatch_size) + value.shape[1:]) for k, value in ordered.items()}
            (params, opt_state, rms), losses = jax.lax.scan(update_minibatch, (params, opt_state, rms), batches)
            return (params, opt_state, rms, ek), losses

        epoch_key, next_key = jax.random.split(next_key)
        (params, opt_state, _final_rms, final_key), losses = jax.lax.scan(
            update_epoch,
            (state.params, state.opt_state, state.obs_rms, epoch_key),
            None,
            length=num_epochs,
        )
        done_flags = jnp.logical_or(data["terminated"].astype(bool), data["truncated"].astype(bool)).astype(jnp.float32)
        done_count = jnp.sum(done_flags)
        metrics = {
            "rollout_return": jnp.mean(jnp.sum(rollout_raw_reward, axis=0)),
            "rollout_reward": jnp.mean(data["raw_reward"]),
            "ep_ret": jnp.where(done_count > 0.0, jnp.sum(data["ep_return"]) / done_count, jnp.nan),
            "ep_len": jnp.where(done_count > 0.0, jnp.sum(data["ep_length"]) / done_count, jnp.nan),
            "done_fraction": jnp.mean(done_flags),
            "loss": jnp.mean(losses[0]),
            "loss_objective": jnp.mean(losses[1]),
            "loss_critic": jnp.mean(losses[2]),
            "entropy": jnp.mean(losses[3]),
            "approx_kl": jnp.mean(losses[4]),
            "clipfrac": jnp.mean(losses[5]),
        }
        return _TrainState(
            iteration,
            params,
            opt_state,
            next_rms,
            next_reward_rms,
            next_discounted_return,
            next_obs,
            next_env_state,
            next_running_return,
            next_running_length,
            final_key,
        ), metrics

    return jax.jit(_train_step)


def _make_eval_step(config: MJXPPOConfig, runtime: _Runtime):
    return make_mjx_eval_step(config, runtime, _eval_action)


def train_mjx_ppo(config: MJXPPOConfig) -> MJXPPOResult:
    runtime = _make_runtime(config)
    state, optimizer = _init_train_state(config, runtime)

    def _full_state_restore(full_state, restored_agent):
        if not isinstance(restored_agent, _AgentState):
            return restored_agent
        return full_state._replace(
            iteration=restored_agent.iteration,
            params=restored_agent.params,
            opt_state=restored_agent.opt_state,
            obs_rms=restored_agent.obs_rms,
            reward_rms=restored_agent.reward_rms,
        )

    return run_mjx_training_loop(
        config=config,
        runtime=runtime,
        state=state,
        train_step=_make_train_step(config, runtime, optimizer),
        eval_step=_make_eval_step(config, runtime),
        result_fn=_result,
        eval_args_fn=_eval_args,
        checkpoint_fn=_checkpoint_fn,
        restore_fn=_full_state_restore,
        algo_name="ppo",
        prefix="MJX_PPO: ",
    )


def register() -> None:
    registry.register_algo("mjx_ppo", MJXPPOConfig, train_mjx_ppo)
