from __future__ import annotations

import time
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Any

import numpy as np

from optimizer.datum import Datum
from optimizer.designer_errors import NoSuchDesignerError
from optimizer.optimizer_types import IterateResult
from optimizer.trajectory import Trajectory


def _require_stack():
    try:
        import jax
        import jax.numpy as jnp
        import optax
        from hyperscalees.models.common import simple_es_tree_key
        from hyperscalees.noiser import all_noisers
    except ImportError as exc:
        raise ImportError(
            "EggRollDesigner requires the separate HyperscaleES environment. "
            "Run admin/setup-hyperscalees.sh first, then use the plain python CLI from that environment."
        ) from exc
    return jax, jnp, optax, simple_es_tree_key, all_noisers


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "t", "1", "yes"}:
            return True
        if lower in {"false", "f", "0", "no"}:
            return False
    raise NoSuchDesignerError(f"EggRoll option '{name}' must be a bool.")


def _as_unit_decay(value: Any, *, name: str) -> float:
    parsed = float(value)
    if parsed <= 0.0 or parsed > 1.0:
        raise NoSuchDesignerError(f"EggRoll option '{name}' must be in the interval (0, 1].")
    return parsed


@dataclass(frozen=True)
class _EggRollDesignerConfig:
    noiser: str = "eggroll"
    sigma: float = 0.05
    lr: float = 0.02
    lr_decay: float = 1.0
    sigma_decay: float = 1.0
    rank: int = 8
    rank_transform: bool = False
    deterministic_policy: bool = False
    steps: int = 200
    eval_episodes: int = 8
    optax: str = "adamw"
    b1: float = 0.9
    b2: float = 0.999
    weight_decay: float = 0.0
    group_size: int = 0
    freeze_nonlora: bool = False
    noise_reuse: int = 0
    use_batched_update: bool = True
    suppress_noiser_stdout: bool = True
    seed_offset: int = 0


def _designer_config(options: dict[str, Any]) -> _EggRollDesignerConfig:
    allowed = set(_EggRollDesignerConfig.__dataclass_fields__)
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise NoSuchDesignerError(f"Unknown EggRoll option(s): {unknown}.")
    return _EggRollDesignerConfig(**options)


def _learning_rate_schedule(jnp, *, base_lr: float, lr_decay: float):
    def schedule(count):
        return jnp.asarray(base_lr, dtype=jnp.float32) * jnp.power(jnp.asarray(lr_decay, dtype=jnp.float32), count)

    return schedule


def _solver(optax_mod, name: str, *, b1: float, b2: float, weight_decay: float):
    solvers = {
        "adam": optax_mod.adam,
        "adamw": optax_mod.adamw,
        "sgd": optax_mod.sgd,
    }
    contrib = getattr(optax_mod, "contrib", None)
    if contrib is not None and hasattr(contrib, "muon"):
        solvers["muon"] = contrib.muon
    if name not in solvers:
        allowed = ", ".join(sorted(solvers))
        raise NoSuchDesignerError(f"EggRoll option 'optax' must be one of: {allowed}.")
    return solvers[name]


def _solver_kwargs(name: str, *, b1: float, b2: float, weight_decay: float) -> dict:
    if name == "sgd":
        return {}
    if name == "muon":
        return {"weight_decay": weight_decay}
    kwargs = {"b1": b1, "b2": b2}
    if name == "adamw":
        kwargs["weight_decay"] = weight_decay
    return kwargs


class EggRollDesigner:
    """JAX HyperscaleES optimizer integrated into the experiment runner."""

    def __init__(
        self,
        policy,
        env_conf,
        **options,
    ) -> None:
        cfg = _designer_config(options)
        if not hasattr(policy, "model_cls") or not hasattr(policy, "params"):
            raise NoSuchDesignerError("Designer 'eggroll' requires an EggRoll policy tag, e.g. policy_tag='eggroll-ac-mlp-64x2-silu'.")
        if env_conf is None:
            raise NoSuchDesignerError("Designer 'eggroll' requires env_conf.")
        env_name = str(getattr(env_conf, "env_name", ""))
        from problems.eggroll_env_adapters import make_eggroll_env_adapter, supports_eggroll_env_adapter

        if not supports_eggroll_env_adapter(env_name):
            raise NoSuchDesignerError(f"Designer 'eggroll' requires a supported EggRoll adapter env tag (got {env_name!r}).")

        jax, jnp, optax_mod, simple_es_tree_key, all_noisers = _require_stack()
        if cfg.noiser not in all_noisers:
            raise NoSuchDesignerError(f"Unknown HyperscaleES noiser '{cfg.noiser}'. Available: {sorted(all_noisers)}")
        solver = _solver(optax_mod, str(cfg.optax), b1=float(cfg.b1), b2=float(cfg.b2), weight_decay=float(cfg.weight_decay))

        self._policy = policy
        self._jax = jax
        self._jnp = jnp
        self._steps_per_episode = int(cfg.steps)
        self._eval_episodes = int(cfg.eval_episodes)
        self._sigma = float(cfg.sigma)
        self._sigma_decay = _as_unit_decay(cfg.sigma_decay, name="sigma_decay")
        self._suppress_noiser_stdout = _as_bool(cfg.suppress_noiser_stdout, name="suppress_noiser_stdout")
        self._best_datum = None
        self._epoch = 0

        if self._sigma <= 0.0:
            raise NoSuchDesignerError("EggRoll option 'sigma' must be > 0.")
        lr = float(cfg.lr)
        if lr <= 0.0:
            raise NoSuchDesignerError("EggRoll option 'lr' must be > 0.")
        if self._steps_per_episode < 1:
            raise NoSuchDesignerError("EggRoll option 'steps' must be >= 1.")
        if self._eval_episodes < 1:
            raise NoSuchDesignerError("EggRoll option 'eval_episodes' must be >= 1.")
        rank = int(cfg.rank)
        if rank < 1:
            raise NoSuchDesignerError("EggRoll option 'rank' must be >= 1.")

        noiser = all_noisers[cfg.noiser]
        lr_decay = _as_unit_decay(cfg.lr_decay, name="lr_decay")
        rank_transform = _as_bool(cfg.rank_transform, name="rank_transform")
        deterministic_policy = _as_bool(cfg.deterministic_policy, name="deterministic_policy")
        seed = (0 if getattr(policy, "problem_seed", None) is None else int(policy.problem_seed)) + int(cfg.seed_offset)
        key = jax.random.key(seed & 0xFFFFFFFF)
        es_key = jax.random.fold_in(key, 1)
        self._train_key = jax.random.fold_in(key, 2)
        self._eval_key = jax.random.fold_in(key, 3)

        env_adapter = make_eggroll_env_adapter(env_name, jax=jax, jnp=jnp)
        frozen_params = policy.frozen_params
        self._params = policy.params
        es_tree_key = simple_es_tree_key(self._params, es_key, policy.scan_map)

        scheduled_lr = _learning_rate_schedule(jnp, base_lr=lr, lr_decay=lr_decay) if lr_decay != 1.0 else lr
        frozen_noiser_params, noiser_params = noiser.init_noiser(
            self._params,
            self._sigma,
            scheduled_lr,
            solver=solver,
            solver_kwargs=_solver_kwargs(str(cfg.optax), b1=float(cfg.b1), b2=float(cfg.b2), weight_decay=float(cfg.weight_decay)),
            group_size=int(cfg.group_size),
            freeze_nonlora=_as_bool(cfg.freeze_nonlora, name="freeze_nonlora"),
            noise_reuse=int(cfg.noise_reuse),
            rank=rank,
            use_batched_update=_as_bool(cfg.use_batched_update, name="use_batched_update"),
        )
        self._noiser_params = noiser_params
        self._evaluate_population, self._evaluate_policy, self._update_params = self._build_jitted_fns(
            env_adapter=env_adapter,
            noiser=noiser,
            frozen_noiser_params=frozen_noiser_params,
            frozen_params=frozen_params,
            es_tree_key=es_tree_key,
            es_map=policy.es_map,
            rank_transform=rank_transform,
            deterministic_policy=deterministic_policy,
        )

    def _build_jitted_fns(
        self,
        *,
        env_adapter,
        noiser,
        frozen_noiser_params,
        frozen_params,
        es_tree_key,
        es_map,
        rank_transform: bool,
        deterministic_policy: bool,
    ):
        jax = self._jax
        jnp = self._jnp
        model_cls = self._policy.model_cls
        steps_per_episode = int(self._steps_per_episode)

        def select_action(policy_dist, action_key):
            if not deterministic_policy:
                return policy_dist.sample(seed=action_key)
            mode = getattr(policy_dist, "mode", None)
            if callable(mode):
                return mode()
            mean = getattr(policy_dist, "mean", None)
            if callable(mean):
                return mean()
            if mean is not None:
                return mean
            return policy_dist.sample(seed=action_key)

        def rollout(params, noiser_params, thread_info, rollout_key):
            reset_key, loop_key = jax.random.split(rollout_key)
            obs, state = env_adapter.reset(reset_key)
            total_reward = jnp.array(0.0, dtype=jnp.float32)
            done = jnp.array(False)

            def step(carry, _unused):
                obs_t, state_t, total_t, done_t, key_t = carry
                key_t, action_key, env_key = jax.random.split(key_t, 3)
                policy_dist = model_cls.forward(
                    noiser,
                    frozen_noiser_params,
                    noiser_params,
                    frozen_params,
                    params,
                    es_tree_key,
                    thread_info,
                    obs_t,
                )
                action = select_action(policy_dist, action_key)
                action = env_adapter.clip_action(action)
                next_obs, next_state, reward, next_done, _info = env_adapter.step(env_key, state_t, action)
                active = jnp.logical_not(done_t)
                obs_out = jax.tree.map(lambda new, old: jnp.where(active, new, old), next_obs, obs_t)
                state_out = jax.tree.map(lambda new, old: jnp.where(active, new, old), next_state, state_t)
                total_out = total_t + jnp.where(active, reward, 0.0)
                done_out = jnp.logical_or(done_t, next_done)
                return (obs_out, state_out, total_out, done_out, key_t), None

            (_, _, total_reward, _, _), _ = jax.lax.scan(step, (obs, state, total_reward, done, loop_key), None, length=steps_per_episode)
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
            population = raw_scores.shape[0]
            iterinfo = (jnp.full((population,), epoch, dtype=jnp.int32), jnp.arange(population, dtype=jnp.int32))
            if rank_transform:
                ranks = jnp.argsort(jnp.argsort(raw_scores)).astype(jnp.float32)
                raw_scores = ranks / jnp.maximum(float(population - 1), 1.0)
            fitnesses = noiser.convert_fitnesses(frozen_noiser_params, noiser_params, raw_scores)
            return noiser.do_updates(frozen_noiser_params, noiser_params, params, es_tree_key, fitnesses, iterinfo, es_map)

        return evaluate_population, evaluate_policy, update_params

    def best_datum(self):
        return self._best_datum

    def _block_tree(self, tree):
        for leaf in self._jax.tree_util.tree_leaves(tree):
            leaf.block_until_ready()
        return tree

    def _current_sigma(self) -> float:
        return self._sigma * (self._sigma_decay**self._epoch)

    def _scheduled_noiser_params(self):
        if not isinstance(self._noiser_params, dict) or "sigma" not in self._noiser_params:
            if self._sigma_decay != 1.0:
                raise NoSuchDesignerError("EggRoll sigma_decay requires noiser_params to expose a 'sigma' field.")
            return self._noiser_params
        return self._noiser_params | {"sigma": self._current_sigma()}

    def iterate(self, _data, num_arms: int, *, telemetry=None) -> IterateResult:
        if int(num_arms) < 2 or int(num_arms) % 2 != 0:
            raise NoSuchDesignerError("EggRoll requires an even population >= 2 because HyperscaleES noisers use mirrored thread pairs.")

        jax = self._jax
        jnp = self._jnp
        epoch = jnp.asarray(self._epoch, dtype=jnp.int32)
        self._noiser_params = self._scheduled_noiser_params()

        t_eval = time.time()
        self._train_key, batch_key = jax.random.split(self._train_key)
        batch_keys = jax.random.split(batch_key, int(num_arms))
        raw_scores = self._evaluate_population(self._params, self._noiser_params, epoch, batch_keys)
        raw_scores = jax.block_until_ready(raw_scores)
        train_eval_dt = time.time() - t_eval

        t_prop = time.time()
        stdout_ctx = redirect_stdout(StringIO()) if self._suppress_noiser_stdout else nullcontext()
        with stdout_ctx:
            self._noiser_params, self._params = self._update_params(self._noiser_params, self._params, raw_scores, epoch)
        self._block_tree(self._params)
        prop_dt = time.time() - t_prop

        t_eval = time.time()
        self._eval_key, eval_batch_key = jax.random.split(self._eval_key)
        eval_scores = self._evaluate_policy(self._params, self._noiser_params, jax.random.split(eval_batch_key, self._eval_episodes))
        eval_scores = jax.block_until_ready(eval_scores)
        policy_eval_dt = time.time() - t_eval

        eval_mean = float(jnp.mean(eval_scores))
        policy = self._policy.with_params(self._params)
        num_steps = int((int(num_arms) + self._eval_episodes) * self._steps_per_episode)
        datum = Datum(
            self,
            policy,
            None,
            Trajectory(
                rreturn=eval_mean,
                states=np.empty((0,)),
                actions=np.empty((0,)),
                num_steps=num_steps,
            ),
        )
        if self._best_datum is None or datum.trajectory.get_decision_rreturn() > self._best_datum.trajectory.get_decision_rreturn():
            self._best_datum = datum

        if telemetry is not None:
            telemetry.set_dt_fit(prop_dt)
            telemetry.set_dt_select(0.0)

        self._epoch += 1
        return IterateResult(data=[datum], dt_prop=float(prop_dt), dt_eval=float(train_eval_dt + policy_eval_dt))
