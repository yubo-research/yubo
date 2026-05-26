from __future__ import annotations

import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from rl import registry
from rl.mjx_ppo_config import MJXPPOConfig


class _TrainState(NamedTuple):
    params: dict[str, Any]
    opt_state: Any
    obs: Any
    env_state: Any
    key: Any


class MJXPPOResult(NamedTuple):
    best_return: float
    last_eval_return: float
    num_iterations: int


class _Runtime(NamedTuple):
    jax: Any
    jnp: Any
    optax: Any
    adapter: Any
    obs_dim: int
    act_dim: int


def _require_stack():
    import jax
    import jax.numpy as jnp
    import optax

    return jax, jnp, optax


def _init_layer(jax, jnp, key, in_dim: int, out_dim: int):
    w_key, _b_key = jax.random.split(key)
    scale = np.sqrt(2.0 / float(in_dim))
    return {"w": jax.random.normal(w_key, (in_dim, out_dim)) * scale, "b": jnp.zeros((out_dim,))}


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


def _normal_log_prob(jnp, raw_action, mean, std):
    z = (raw_action - mean) / std
    log_prob = -0.5 * z * z - jnp.log(std) - 0.5 * jnp.log(2.0 * jnp.pi)
    squash_correction = jnp.log(1.0 - jnp.tanh(raw_action) ** 2 + 1e-6)
    return jnp.sum(log_prob - squash_correction, axis=-1)


def _sample_action(jax, jnp, params, obs, key):
    mean, std = _policy(params, jnp, obs)
    raw = mean + std * jax.random.normal(key, mean.shape)
    return jnp.tanh(raw), _normal_log_prob(jnp, raw, mean, std), raw


def _make_runtime(config: MJXPPOConfig):
    jax, jnp, optax = _require_stack()
    from problems.jax_env_factory import make_jax_env_adapter

    adapter = make_jax_env_adapter(config.env_tag, jax=jax, jnp=jnp)
    obs_dim = int(np.prod(tuple(adapter.observation_space.shape)))
    act_dim = int(np.prod(tuple(adapter.action_space.shape)))
    return _Runtime(jax, jnp, optax, adapter, obs_dim, act_dim)


def _make_train_step(config: MJXPPOConfig, adapter, optimizer):
    jax, jnp, _optax = _require_stack()
    num_envs = int(config.collector.num_envs)
    num_steps = int(config.collector.num_steps)
    minibatch_size = int(config.optim.minibatch_size)
    num_epochs = int(config.optim.num_epochs)

    def rollout(params, obs, env_state, key):
        def step(carry, _):
            obs_t, state_t, key_t = carry
            key_t, act_key, env_key = jax.random.split(key_t, 3)
            action, log_prob, raw_action = _sample_action(jax, jnp, params, obs_t, act_key)
            keys = jax.random.split(env_key, num_envs)
            next_obs, next_state, reward, done, _info = jax.vmap(adapter.step)(keys, state_t, action)
            transition = {
                "obs": obs_t,
                "raw_action": raw_action,
                "log_prob": log_prob,
                "reward": reward,
                "done": done,
                "value": _value(params, jnp, obs_t),
            }
            return (next_obs, next_state, key_t), transition

        return jax.lax.scan(step, (obs, env_state, key), None, length=num_steps)

    def advantages(params, last_obs, data):
        last_value = _value(params, jnp, last_obs)

        def step(carry, transition):
            gae, next_value = carry
            mask = 1.0 - transition["done"]
            delta = transition["reward"] + config.loss.gamma * next_value * mask - transition["value"]
            gae = delta + config.loss.gamma * config.loss.gae_lambda * mask * gae
            return (gae, transition["value"]), gae

        _carry, adv = jax.lax.scan(step, (jnp.zeros_like(last_value), last_value), data, reverse=True)
        return adv, adv + data["value"]

    def loss_fn(params, batch):
        mean, std = _policy(params, jnp, batch["obs"])
        log_prob = _normal_log_prob(jnp, batch["raw_action"], mean, std)
        ratio = jnp.exp(log_prob - batch["log_prob"])
        clipped = jnp.clip(ratio, 1.0 - config.loss.clip_epsilon, 1.0 + config.loss.clip_epsilon)
        policy_loss = -jnp.mean(jnp.minimum(ratio * batch["adv"], clipped * batch["adv"]))
        value_loss = jnp.mean((_value(params, jnp, batch["obs"]) - batch["target"]) ** 2)
        entropy = jnp.mean(jnp.sum(jnp.log(std) + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1))
        loss = policy_loss + config.loss.critic_coeff * value_loss - config.loss.entropy_coeff * entropy
        approx_kl = jnp.mean(batch["log_prob"] - log_prob)
        clipfrac = jnp.mean((jnp.abs(ratio - 1.0) > config.loss.clip_epsilon).astype(jnp.float32))
        return loss, (policy_loss, value_loss, entropy, approx_kl, clipfrac)

    def update_minibatch(carry, mb):
        params, opt_state = carry
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, mb)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = _optax.apply_updates(params, updates)
        return (params, opt_state), (loss, *aux)

    def train_step(state: _TrainState):
        (next_obs, next_env_state, key), data = rollout(state.params, state.obs, state.env_state, state.key)
        adv, target = advantages(state.params, next_obs, data)
        flat = {key: value.reshape((-1,) + value.shape[2:]) for key, value in data.items()}
        flat["adv"] = ((adv - adv.mean()) / (adv.std() + 1e-8)).reshape(-1)
        flat["target"] = target.reshape(-1)
        usable = (flat["adv"].shape[0] // minibatch_size) * minibatch_size
        next_key = key

        def update_epoch(carry, _):
            params, opt_state, epoch_key = carry
            epoch_key, perm_key = jax.random.split(epoch_key)
            order = jax.random.permutation(perm_key, flat["adv"].shape[0])
            ordered = {key: value[order] for key, value in flat.items()}
            batches = {key: value[:usable].reshape((-1, minibatch_size) + value.shape[1:]) for key, value in ordered.items()}
            (params, opt_state), losses = jax.lax.scan(update_minibatch, (params, opt_state), batches)
            return (params, opt_state, epoch_key), losses

        (params, opt_state, next_key), losses = jax.lax.scan(update_epoch, (state.params, state.opt_state, next_key), None, length=num_epochs)
        metrics = {
            "eval_return": jnp.mean(jnp.sum(data["reward"], axis=0)),
            "loss": jnp.mean(losses[0]),
            "loss_objective": jnp.mean(losses[1]),
            "loss_critic": jnp.mean(losses[2]),
            "entropy": jnp.mean(losses[3]),
            "approx_kl": jnp.mean(losses[4]),
            "clipfrac": jnp.mean(losses[5]),
        }
        return _TrainState(params, opt_state, next_obs, next_env_state, next_key), metrics

    return jax.jit(train_step)


def train_mjx_ppo(config: MJXPPOConfig) -> MJXPPOResult:
    runtime = _make_runtime(config)
    jax, jnp, optax = runtime.jax, runtime.jnp, runtime.optax
    adapter, obs_dim, act_dim = runtime.adapter, runtime.obs_dim, runtime.act_dim
    key = jax.random.key(int(config.seed))
    key, reset_key, param_key = jax.random.split(key, 3)
    reset_keys = jax.random.split(reset_key, int(config.collector.num_envs))
    obs, env_state = jax.vmap(adapter.reset)(reset_keys)
    params = _init_params(jax, jnp, param_key, obs_dim, act_dim, int(config.hidden_size))
    optimizer = optax.chain(optax.clip_by_global_norm(float(config.optim.max_grad_norm)), optax.adam(float(config.optim.lr)))
    state = _TrainState(params, optimizer.init(params), obs, env_state, key)
    train_step = _make_train_step(config, adapter, optimizer)
    iterations = int(config.collector.total_frames) // (int(config.collector.num_envs) * int(config.collector.num_steps))
    exp_dir = Path(config.exp_dir) / f"seed_{int(config.seed)}"
    metrics_path = exp_dir / "metrics.jsonl"
    exp_dir.mkdir(parents=True, exist_ok=True)

    from rl import logger

    start = time.time()
    best_return = float("-inf")
    last_return = float("nan")
    print(
        f"[rl/ppo/mjx] env_tag={config.env_tag} seed={config.seed} obs_dim={obs_dim} act_dim={act_dim} num_envs={config.collector.num_envs} num_steps={config.collector.num_steps} iters={iterations}",
        flush=True,
    )
    logger.log_run_header_basic(
        algo_name="ppo",
        env_tag=config.env_tag,
        seed=int(config.seed),
        backbone_name="jax-mlp",
        from_pixels=False,
        obs_dim=obs_dim,
        act_dim=act_dim,
        frames_per_batch=int(config.collector.num_envs) * int(config.collector.num_steps),
        num_iterations=iterations,
        device_type=str(jax.default_backend()),
        config_obj=config,
    )
    for iteration in range(1, iterations + 1):
        state, metrics = train_step(state)
        metrics_host = {key: float(value) for key, value in jax.device_get(metrics).items()}
        last_return = metrics_host["eval_return"]
        best_return = max(best_return, last_return)
        if iteration % int(config.log_interval) == 0 or iteration == iterations:
            elapsed = time.time() - start
            global_step = iteration * int(config.collector.num_envs) * int(config.collector.num_steps)
            record = {
                "iteration": iteration,
                "global_step": global_step,
                "eval_return": last_return,
                "best_return": best_return,
                "approx_kl": metrics_host["approx_kl"],
                "clipfrac": metrics_host["clipfrac"],
                "loss": metrics_host["loss"],
                "loss_objective": metrics_host["loss_objective"],
                "loss_critic": metrics_host["loss_critic"],
                "entropy": metrics_host["entropy"],
                "time_seconds": elapsed,
            }
            logger.append_metrics(metrics_path, record)
            logger.log_progress_iteration(
                iteration,
                iterations,
                int(config.collector.num_envs) * int(config.collector.num_steps),
                elapsed,
                eval_return=last_return,
                best_return=best_return,
                algo_metrics={"kl": metrics_host["approx_kl"], "clipfrac": metrics_host["clipfrac"]},
                algo_name="ppo",
            )
    logger.log_run_footer(best_return, iterations, time.time() - start, algo_name="ppo")
    return MJXPPOResult(best_return=best_return, last_eval_return=last_return, num_iterations=iterations)


def register() -> None:
    registry.register_algo("mjx_ppo", MJXPPOConfig, train_mjx_ppo)
