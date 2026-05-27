from __future__ import annotations

import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from problems.jax_env_core import (
    _init_reward_rms,
    _init_rms,
    _normalize_obs,
    _normalize_reward,
    _RewardRunningMeanStd,
    _RunningMeanStd,
    _update_reward_rms,
    _update_rms,
)
from rl import registry
from rl.mjx_ppo import (
    _init_layer,
    _linear,
    _make_runtime,
    _normal_log_prob,
    _policy,
    _Runtime,
)
from rl.mjx_sac_config import MJXSACConfig


class _Replay(NamedTuple):
    obs: Any
    action: Any
    reward: Any
    done: Any
    next_obs: Any
    ptr: Any
    size: Any


class _TrainState(NamedTuple):
    actor: dict[str, Any]
    critic1: dict[str, Any]
    critic2: dict[str, Any]
    target1: dict[str, Any]
    target2: dict[str, Any]
    actor_opt: Any
    critic_opt: Any
    alpha_opt: Any
    log_alpha: Any
    obs_rms: _RunningMeanStd
    reward_rms: _RewardRunningMeanStd
    discounted_return: Any
    obs: Any
    env_state: Any
    replay: _Replay
    key: Any


class MJXSACResult(NamedTuple):
    best_return: float
    last_rollout_return: float
    num_steps: int


def _init_q(jax, jnp, key, obs_dim: int, act_dim: int, hidden: int):
    keys = jax.random.split(key, 3)
    return {
        "q1": _init_layer(jax, jnp, keys[0], obs_dim + act_dim, hidden),
        "q2": _init_layer(jax, jnp, keys[1], hidden, hidden),
        "qout": _init_layer(jax, jnp, keys[2], hidden, 1),
    }


def _q_value(params, jnp, obs, action):
    x = jnp.concatenate([obs, action], axis=-1)
    h = jnp.tanh(_linear(params["q1"], x))
    h = jnp.tanh(_linear(params["q2"], h))
    return jnp.squeeze(_linear(params["qout"], h), axis=-1)


def _init_actor(jax, jnp, key, obs_dim: int, act_dim: int, hidden: int):
    keys = jax.random.split(key, 3)
    return {
        "policy_1": _init_layer(jax, jnp, keys[0], obs_dim, hidden),
        "policy_2": _init_layer(jax, jnp, keys[1], hidden, hidden),
        "policy_mean": _init_layer(jax, jnp, keys[2], hidden, act_dim),
        "log_std": jnp.full((act_dim,), -0.5),
    }


def _sample_action(jax, jnp, actor, obs, key):
    mean, std = _policy(actor, jnp, obs)
    raw = mean + std * jax.random.normal(key, mean.shape)
    return jnp.tanh(raw), _normal_log_prob(jnp, raw, mean, std)


def _init_replay(jnp, capacity: int, obs_dim: int, act_dim: int):
    return _Replay(
        obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
        action=jnp.zeros((capacity, act_dim), dtype=jnp.float32),
        reward=jnp.zeros((capacity,), dtype=jnp.float32),
        done=jnp.zeros((capacity,), dtype=jnp.float32),
        next_obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
        ptr=jnp.array(0, dtype=jnp.int32),
        size=jnp.array(0, dtype=jnp.int32),
    )


def _insert_replay(jnp, replay: _Replay, data: dict[str, Any], capacity: int) -> _Replay:
    count = data["reward"].shape[0]
    idx = (jnp.arange(count, dtype=jnp.int32) + replay.ptr) % int(capacity)
    return _Replay(
        obs=replay.obs.at[idx].set(data["obs"]),
        action=replay.action.at[idx].set(data["action"]),
        reward=replay.reward.at[idx].set(data["reward"]),
        done=replay.done.at[idx].set(data["done"]),
        next_obs=replay.next_obs.at[idx].set(data["next_obs"]),
        ptr=(replay.ptr + count) % int(capacity),
        size=jnp.minimum(replay.size + count, int(capacity)),
    )


def _make_train_step(config: MJXSACConfig, runtime: _Runtime, optimizers):
    jax, jnp, optax = runtime.jax, runtime.jnp, runtime.optax
    actor_optim, critic_optim, alpha_optim = optimizers
    num_envs = int(config.collector.num_envs)
    num_steps = int(config.collector.num_steps)
    batch_size = int(config.collector.batch_size)
    replay_size = int(config.collector.replay_size)
    updates_per_iter = int(config.collector.updates_per_iter)
    target_entropy = -float(runtime.act_dim) if config.loss.target_entropy is None else float(config.loss.target_entropy)

    def rollout(state: _TrainState):
        def step(carry, _):
            obs_t, env_state_t, key_t, rms_t, r_rms_t, disc_ret_t = carry
            key_t, act_key, env_key = jax.random.split(key_t, 3)

            # Normalize observation before policy/actor
            norm_obs = _normalize_obs(obs_t, rms_t, jnp)
            action, _log_prob = _sample_action(jax, jnp, state.actor, norm_obs, act_key)

            keys = jax.random.split(env_key, num_envs)
            next_obs, next_env_state, reward, done, _info = jax.vmap(runtime.adapter.step)(keys, env_state_t, action)

            # Update Discounted Return (Standard way to normalize rewards)
            disc_ret_t = reward + config.loss.gamma * disc_ret_t * (1.0 - done)

            # Update RMS statistics
            rms_t = _update_rms(rms_t, next_obs, jnp)
            r_rms_t = _update_reward_rms(r_rms_t, disc_ret_t, jnp)

            transition = {
                "obs": obs_t,
                "action": action,
                "reward": reward,
                "done": done,
                "next_obs": next_obs,
            }
            return (
                next_obs,
                next_env_state,
                key_t,
                rms_t,
                r_rms_t,
                disc_ret_t,
            ), transition

        return jax.lax.scan(
            step,
            (
                state.obs,
                state.env_state,
                state.key,
                state.obs_rms,
                state.reward_rms,
                state.discounted_return,
            ),
            None,
            length=num_steps,
        )

    def sample(replay: _Replay, key):
        idx = jax.random.randint(key, (batch_size,), 0, replay.size)
        return {
            "obs": replay.obs[idx],
            "action": replay.action[idx],
            "reward": replay.reward[idx],
            "done": replay.done[idx],
            "next_obs": replay.next_obs[idx],
        }

    def critic_loss(
        critic1,
        critic2,
        actor,
        target1,
        target2,
        log_alpha,
        obs_rms,
        reward_rms,
        batch,
        key,
    ):
        # Normalize observations and rewards on-the-fly during update
        obs = _normalize_obs(batch["obs"], obs_rms, jnp)
        next_obs = _normalize_obs(batch["next_obs"], obs_rms, jnp)
        reward = _normalize_reward(batch["reward"], reward_rms, jnp)

        next_action, next_log_prob = _sample_action(jax, jnp, actor, next_obs, key)
        target_q = jnp.minimum(
            _q_value(target1, jnp, next_obs, next_action),
            _q_value(target2, jnp, next_obs, next_action),
        )
        alpha = jnp.exp(log_alpha)
        y = reward + config.loss.gamma * (1.0 - batch["done"]) * (target_q - alpha * next_log_prob)
        q1 = _q_value(critic1, jnp, obs, batch["action"])
        q2 = _q_value(critic2, jnp, obs, batch["action"])
        return jnp.mean((q1 - y) ** 2 + (q2 - y) ** 2)

    def actor_loss(actor, critic1, critic2, log_alpha, rms, batch, key):
        obs = _normalize_obs(batch["obs"], rms, jnp)
        action, log_prob = _sample_action(jax, jnp, actor, obs, key)
        q = jnp.minimum(_q_value(critic1, jnp, obs, action), _q_value(critic2, jnp, obs, action))
        alpha = jnp.exp(log_alpha)
        return jnp.mean(alpha * log_prob - q)

    def alpha_loss(actor, log_alpha, rms, batch, key):
        obs = _normalize_obs(batch["obs"], rms, jnp)
        _action, log_prob = _sample_action(jax, jnp, actor, obs, key)
        return -jnp.mean(log_alpha * jax.lax.stop_gradient(log_prob + target_entropy))

    def soft_update(params, target):
        return jax.tree_util.tree_map(
            lambda p, t: config.loss.tau * p + (1.0 - config.loss.tau) * t,
            params,
            target,
        )

    def update_once(state: _TrainState, _):
        key, sample_key, critic_key, actor_key = jax.random.split(state.key, 4)
        batch = sample(state.replay, sample_key)
        critic_grad_fn = jax.value_and_grad(critic_loss, argnums=(0, 1))
        critic_loss_value, critic_grads = critic_grad_fn(
            state.critic1,
            state.critic2,
            state.actor,
            state.target1,
            state.target2,
            state.log_alpha,
            state.obs_rms,
            state.reward_rms,
            batch,
            critic_key,
        )
        critic_updates, critic_opt = critic_optim.update(critic_grads, state.critic_opt, (state.critic1, state.critic2))
        critic1, critic2 = optax.apply_updates((state.critic1, state.critic2), critic_updates)
        actor_loss_value, actor_grad = jax.value_and_grad(actor_loss)(
            state.actor,
            critic1,
            critic2,
            state.log_alpha,
            state.obs_rms,
            batch,
            actor_key,
        )
        alpha_loss_value, alpha_grad = jax.value_and_grad(alpha_loss, argnums=1)(state.actor, state.log_alpha, state.obs_rms, batch, actor_key)
        actor_updates, actor_opt = actor_optim.update(actor_grad, state.actor_opt, state.actor)
        alpha_updates, alpha_opt = alpha_optim.update(alpha_grad, state.alpha_opt, state.log_alpha)
        actor = optax.apply_updates(state.actor, actor_updates)
        log_alpha = optax.apply_updates(state.log_alpha, alpha_updates)
        next_state = state._replace(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            target1=soft_update(critic1, state.target1),
            target2=soft_update(critic2, state.target2),
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            alpha_opt=alpha_opt,
            log_alpha=log_alpha,
            key=key,
        )
        return next_state, (
            actor_loss_value,
            critic_loss_value,
            alpha_loss_value,
            jnp.exp(log_alpha),
        )

    def train_step(state: _TrainState):
        (obs, env_state, key, obs_rms, reward_rms, discounted_return), data = rollout(state)
        flat = {name: value.reshape((-1,) + value.shape[2:]) for name, value in data.items()}
        replay = _insert_replay(jnp, state.replay, flat, replay_size)
        state = state._replace(
            obs=obs,
            env_state=env_state,
            obs_rms=obs_rms,
            reward_rms=reward_rms,
            discounted_return=discounted_return,
            replay=replay,
            key=key,
        )
        state, losses = jax.lax.scan(update_once, state, None, length=updates_per_iter)
        metrics = {
            "rollout_return": jnp.mean(jnp.sum(data["reward"], axis=0)),
            "rollout_reward": jnp.mean(data["reward"]),
            "done_fraction": jnp.mean(data["done"]),
            "loss_actor": jnp.mean(losses[0]),
            "loss_critic": jnp.mean(losses[1]),
            "loss_alpha": jnp.mean(losses[2]),
            "alpha_value": jnp.mean(losses[3]),
        }
        return state, metrics

    return jax.jit(train_step)


def _init_state(config: MJXSACConfig, runtime: _Runtime):
    jax, jnp, optax = runtime.jax, runtime.jnp, runtime.optax
    key = jax.random.key(int(config.seed))
    key, reset_key, actor_key, q1_key, q2_key = jax.random.split(key, 5)
    reset_keys = jax.random.split(reset_key, int(config.collector.num_envs))
    obs, env_state = jax.vmap(runtime.adapter.reset)(reset_keys)
    actor = _init_actor(jax, jnp, actor_key, runtime.obs_dim, runtime.act_dim, int(config.hidden_size))
    critic1 = _init_q(jax, jnp, q1_key, runtime.obs_dim, runtime.act_dim, int(config.hidden_size))
    critic2 = _init_q(jax, jnp, q2_key, runtime.obs_dim, runtime.act_dim, int(config.hidden_size))
    actor_optim = optax.adam(float(config.optim.lr_actor))
    critic_optim = optax.adam(float(config.optim.lr_critic))
    alpha_optim = optax.adam(float(config.optim.lr_alpha))
    log_alpha = jnp.asarray(np.log(float(config.loss.alpha_init)), dtype=jnp.float32)
    replay = _init_replay(jnp, int(config.collector.replay_size), runtime.obs_dim, runtime.act_dim)

    # Initialize RMS
    obs_rms = _init_rms(jnp, (runtime.obs_dim,))
    reward_rms = _init_reward_rms(jnp)
    discounted_return = jnp.zeros((int(config.collector.num_envs),), dtype=jnp.float32)

    state = _TrainState(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        target1=critic1,
        target2=critic2,
        actor_opt=actor_optim.init(actor),
        critic_opt=critic_optim.init((critic1, critic2)),
        alpha_opt=alpha_optim.init(log_alpha),
        log_alpha=log_alpha,
        obs_rms=obs_rms,
        reward_rms=reward_rms,
        discounted_return=discounted_return,
        obs=obs,
        env_state=env_state,
        replay=replay,
        key=key,
    )
    return state, (actor_optim, critic_optim, alpha_optim)


def train_mjx_sac(config: MJXSACConfig) -> MJXSACResult:
    runtime = _make_runtime(config)
    state, optimizers = _init_state(config, runtime)
    train_step = _make_train_step(config, runtime, optimizers)
    frames_per_iter = int(config.collector.num_envs) * int(config.collector.num_steps)
    iterations = int(config.collector.total_frames) // frames_per_iter
    exp_dir = Path(config.exp_dir) / f"seed_{int(config.seed)}"
    metrics_path = exp_dir / "metrics.jsonl"
    exp_dir.mkdir(parents=True, exist_ok=True)
    from rl import logger

    start = time.time()
    best_return = float("-inf")
    last_rollout_return = float("nan")
    print(
        f"[rl/sac/mjx] env_tag={config.env_tag} seed={config.seed} obs_dim={runtime.obs_dim} act_dim={runtime.act_dim} num_envs={config.collector.num_envs} num_steps={config.collector.num_steps} iters={iterations}",
        flush=True,
    )
    logger.log_run_header_basic(
        algo_name="sac",
        env_tag=config.env_tag,
        seed=int(config.seed),
        backbone_name="jax-mlp",
        from_pixels=False,
        obs_dim=runtime.obs_dim,
        act_dim=runtime.act_dim,
        frames_per_batch=frames_per_iter,
        num_iterations=iterations,
        device_type=str(runtime.jax.default_backend()),
        config_obj=config,
        eval_label="rollout",
    )
    for iteration in range(1, iterations + 1):
        state, metrics = train_step(state)
        values = {name: float(value) for name, value in runtime.jax.device_get(metrics).items()}
        last_rollout_return = values["rollout_return"]
        best_return = max(best_return, last_rollout_return)
        if iteration % int(config.log_interval) == 0 or iteration == iterations:
            elapsed = time.time() - start
            step = iteration * frames_per_iter
            logger.append_metrics(
                metrics_path,
                {
                    "step": step,
                    "eval_return": None,
                    "rollout_return": last_rollout_return,
                    "rollout_reward": values["rollout_reward"],
                    "done_fraction": values["done_fraction"],
                    "best_return": best_return,
                    "loss_actor": values["loss_actor"],
                    "loss_critic": values["loss_critic"],
                    "loss_alpha": values["loss_alpha"],
                    "alpha_value": values["alpha_value"],
                    "time_seconds": elapsed,
                },
            )
            logger.log_eval_iteration(
                0,
                0,
                frames_per_iter,
                eval_return=last_rollout_return,
                best_return=best_return,
                algo_metrics={
                    "actor": values["loss_actor"],
                    "critic": values["loss_critic"],
                    "alpha": values["alpha_value"],
                },
                algo_name="sac",
                elapsed=elapsed,
                step_override=step,
            )
    logger.log_run_footer(
        best_return,
        iterations * frames_per_iter,
        time.time() - start,
        algo_name="sac",
        step_label="steps",
    )
    return MJXSACResult(
        best_return=best_return,
        last_rollout_return=last_rollout_return,
        num_steps=iterations * frames_per_iter,
    )


def register() -> None:
    registry.register_algo("mjx_sac", MJXSACConfig, train_mjx_sac)
