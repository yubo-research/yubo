from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np


_DEFAULT_STACK_ERROR = (
    "EggRoll JAX vector evaluation requires the separate HyperscaleES environment. "
    "Run admin/setup-hyperscalees.sh first, then use the plain python CLI from that environment."
)


def require_eggroll_jax_stack(message: str | None = None):
    try:
        import jax
        import jax.numpy as jnp
        from hyperscalees.models.common import simple_es_tree_key
    except ImportError as exc:
        raise ImportError(_DEFAULT_STACK_ERROR if message is None else message) from exc
    return jax, jnp, simple_es_tree_key


def as_bool(value: Any, *, name: str, error_cls=ValueError, option_label: str = "EggRoll JAX option") -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "t", "1", "yes"}:
            return True
        if lower in {"false", "f", "0", "no"}:
            return False
    raise error_cls(f"{option_label} '{name}' must be a bool.")


class IdentityNoiser:
    @staticmethod
    def get_noisy_standard(_frozen_noiser_params, _noiser_params, params, _es_tree_key, _iterinfo):
        return params

    @staticmethod
    def do_mm(_frozen_noiser_params, _noiser_params, params, _es_tree_key, _iterinfo, x):
        return x @ params.T

    @staticmethod
    def do_Tmm(_frozen_noiser_params, _noiser_params, params, _es_tree_key, _iterinfo, x):
        return x @ params

    @staticmethod
    def do_emb(_frozen_noiser_params, _noiser_params, params, _es_tree_key, _iterinfo, x):
        return params[x]


class EggRollParamCodec:
    def __init__(self, jax, jnp, params_init) -> None:
        self._jax = jax
        self._jnp = jnp
        leaves, treedef = jax.tree_util.tree_flatten(params_init)
        self._leaves_init = tuple(leaves)
        self._treedef = treedef
        self.shapes = tuple(tuple(int(v) for v in leaf.shape) for leaf in leaves)
        self.dtypes = tuple(leaf.dtype for leaf in leaves)
        self.sizes = tuple(int(leaf.size) for leaf in leaves)
        self.offsets = tuple(np.cumsum((0,) + self.sizes[:-1]).astype(np.int64).tolist())
        self.dim = int(sum(self.sizes))
        self.x0 = self.flatten(params_init)

    def flatten(self, params) -> np.ndarray:
        leaves, _treedef = self._jax.tree_util.tree_flatten(params)
        flat = [np.asarray(leaf).reshape(-1).astype(np.float64) for leaf in leaves]
        if not flat:
            return np.empty((0,), dtype=np.float64)
        return np.concatenate(flat, axis=0)

    def decode_absolute(self, x):
        leaves = []
        start = 0
        for shape, dtype, size in zip(self.shapes, self.dtypes, self.sizes, strict=True):
            leaf = self._jnp.reshape(x[start : start + size], shape).astype(dtype)
            leaves.append(leaf)
            start += size
        return self._jax.tree_util.tree_unflatten(self._treedef, leaves)

    def decode_offset(self, x, *, scale: float):
        leaves = []
        start = 0
        for init_leaf, shape, dtype, size in zip(
            self._leaves_init,
            self.shapes,
            self.dtypes,
            self.sizes,
            strict=True,
        ):
            raw = self._jnp.reshape(x[start : start + size], shape).astype(dtype)
            leaves.append(init_leaf + raw * float(scale))
            start += size
        return self._jax.tree_util.tree_unflatten(self._treedef, leaves)


class EggRollActionSelector:
    def __init__(self, jax, jnp, *, deterministic_policy: bool) -> None:
        self._jax = jax
        self._jnp = jnp
        self._deterministic_policy = bool(deterministic_policy)

    def select_action(self, policy_dist, action_key):
        if isinstance(policy_dist, tuple):
            policy_dist = policy_dist[0]
        if not self._deterministic_policy:
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

    def distribution_features(self, policy_dist):
        if isinstance(policy_dist, tuple):
            policy_dist = policy_dist[0]
        logits = getattr(policy_dist, "logits", None)
        if callable(logits):
            logits = logits()
        if logits is not None:
            return self._jnp.ravel(self._jnp.asarray(logits, dtype=self._jnp.float32))
        mean = getattr(policy_dist, "mean", None)
        if callable(mean):
            return self._jnp.ravel(self._jnp.asarray(mean(), dtype=self._jnp.float32))
        if mean is not None:
            return self._jnp.ravel(self._jnp.asarray(mean, dtype=self._jnp.float32))
        probs = getattr(policy_dist, "probs", None)
        if callable(probs):
            probs = probs()
        if probs is not None:
            return self._jnp.ravel(self._jnp.asarray(probs, dtype=self._jnp.float32))
        mode = getattr(policy_dist, "mode", None)
        if callable(mode):
            return self._jnp.ravel(self._jnp.asarray(mode(), dtype=self._jnp.float32))
        return self._jnp.ravel(self._jnp.asarray(policy_dist.sample(seed=self._jax.random.key(0)), dtype=self._jnp.float32))


class EggRollRuntimeEvaluator:
    def __init__(self, runtime: EggRollJAXRuntime) -> None:
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
                action = env_adapter.clip_action(action)
                next_obs, next_state, reward, next_done, _info = env_adapter.step(env_key, state_t, action)
                active = jnp.logical_not(done_t)
                obs_out = jax.tree.map(lambda new, old: jnp.where(active, new, old), next_obs, obs_t)
                state_out = jax.tree.map(lambda new, old: jnp.where(active, new, old), next_state, state_t)
                total_out = total_t + jnp.where(active, reward, 0.0)
                done_out = jnp.logical_or(done_t, next_done)
                return (obs_out, state_out, total_out, done_out, key_t), None

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
            se = jnp.where(scores.shape[0] > 1, jnp.std(scores) / jnp.sqrt(scores.shape[0]), jnp.array(0.0, dtype=jnp.float32))
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
            int(num_candidates) * rt.eval_episodes,
        )
        return keys.reshape((int(num_candidates), rt.eval_episodes))

    def next_eval_keys(self, num_candidates: int):
        self._stateful_eval_key, batch_key = self._runtime.jax.random.split(self._stateful_eval_key)
        keys = self._runtime.jax.random.split(batch_key, int(num_candidates) * self._runtime.eval_episodes)
        return keys.reshape((int(num_candidates), self._runtime.eval_episodes))

    def evaluate(self, x: np.ndarray, *, seed: int) -> tuple[float, float]:
        means, ses = self.evaluate_many(np.asarray([x], dtype=np.float64), seed=int(seed))
        return float(means[0]), float(ses[0])

    def evaluate_many(self, x_batch: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        x_batch = np.asarray(x_batch, dtype=np.float64)
        return self.evaluate_many_with_keys(x_batch, self.keys_for_seed(int(seed), int(x_batch.shape[0])))

    def evaluate_many_with_keys(self, x_batch: np.ndarray, keys_batch) -> tuple[np.ndarray, np.ndarray]:
        rt = self._runtime
        x_batch = np.asarray(x_batch, dtype=np.float64)
        if x_batch.ndim != 2 or x_batch.shape[1] != rt.dim:
            raise ValueError(f"x_batch must have shape (n, {rt.dim}), got {x_batch.shape}.")
        means, ses = self._evaluate_batch_jit(rt.jnp.asarray(x_batch, dtype=rt.jnp.float32), keys_batch)
        means = np.asarray(rt.jax.block_until_ready(means), dtype=np.float64)
        ses = np.asarray(rt.jax.block_until_ready(ses), dtype=np.float64)
        return means, ses

    def evaluate_values_with_keys(self, x_batch: np.ndarray, keys_batch) -> np.ndarray:
        means, _ses = self.evaluate_many_with_keys(x_batch, keys_batch)
        return means


class EggRollRuntimeEmbedder:
    def __init__(self, runtime: EggRollJAXRuntime) -> None:
        self._runtime = runtime
        self._probe_obs = None
        self._embed_batch_jit = None

    def configure(self, num_probes: int) -> None:
        if int(num_probes) < 1:
            raise ValueError("num_probes must be >= 1.")
        keys = self._runtime.jax.random.split(self._runtime.embed_key_base, int(num_probes))
        self._probe_obs = self._runtime.jax.vmap(lambda k: self._runtime.env_adapter.reset(k)[0])(keys)
        self._embed_batch_jit = self._build()

    def _build(self):
        rt = self._runtime
        jax = rt.jax
        jnp = rt.jnp
        model_cls = rt.policy.model_cls
        noiser = rt.identity_noiser
        frozen_params = rt.policy.frozen_params
        es_tree_key = rt.es_tree_key
        probe_obs = self._probe_obs

        def embed_candidate(x):
            params = rt.decode_vector_params(x)

            def features_for_obs(obs):
                policy_dist = model_cls.forward(noiser, None, None, frozen_params, params, es_tree_key, None, obs)
                return rt.action_selector.distribution_features(policy_dist)

            z = jax.vmap(features_for_obs)(probe_obs)
            return jnp.ravel(z)

        @jax.jit
        def embed_batch(x_batch):
            return jax.vmap(embed_candidate)(x_batch)

        return embed_batch

    def embed_many(self, x_batch: np.ndarray) -> np.ndarray:
        if self._embed_batch_jit is None:
            raise ValueError("Behavior embedding is not configured.")
        x_batch = np.asarray(x_batch, dtype=np.float64)
        z = self._embed_batch_jit(self._runtime.jnp.asarray(x_batch, dtype=self._runtime.jnp.float32))
        return np.asarray(self._runtime.jax.block_until_ready(z), dtype=np.float64)

    def embed(self, x: np.ndarray) -> np.ndarray:
        return self.embed_many(np.asarray([x], dtype=np.float64))[0]


class EggRollNoiseSampler:
    def __init__(self, codec: EggRollParamCodec) -> None:
        self._codec = codec

    def sample(
        self,
        *,
        seed: int,
        num_dim_target: float | None = None,
        num_module_target: float | None = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        if num_module_target is not None:
            return self._sample_module_noise(rng, float(num_module_target))
        if num_dim_target is not None:
            return self._sample_dim_noise(rng, float(num_dim_target))
        return rng.standard_normal(self._codec.dim).astype(np.float64)

    def _sample_dim_noise(self, rng: np.random.Generator, target: float) -> np.ndarray:
        if target <= 0:
            raise ValueError("dim perturb target must be > 0.")
        if 0 < target < 1:
            mask = rng.random(self._codec.dim) < target
            if not np.any(mask):
                mask[int(rng.integers(self._codec.dim))] = True
            noise = rng.standard_normal(self._codec.dim).astype(np.float64)
            noise[~mask] = 0.0
            return noise
        k = min(max(int(target), 1), self._codec.dim)
        idx = rng.choice(self._codec.dim, size=k, replace=False)
        noise = np.zeros(self._codec.dim, dtype=np.float64)
        noise[idx] = rng.standard_normal(k)
        return noise

    def _sample_module_noise(self, rng: np.random.Generator, target: float) -> np.ndarray:
        if target <= 0:
            raise ValueError("module perturb target must be > 0.")
        num_leaves = len(self._codec.sizes)
        if 0 < target < 1:
            mask = rng.random(num_leaves) < target
            if not np.any(mask):
                mask[int(rng.integers(num_leaves))] = True
        else:
            k = min(max(int(target), 1), num_leaves)
            mask = np.zeros(num_leaves, dtype=bool)
            mask[rng.choice(num_leaves, size=k, replace=False)] = True
        noise = np.zeros(self._codec.dim, dtype=np.float64)
        for selected, start, size in zip(mask, self._codec.offsets, self._codec.sizes, strict=True):
            if not selected:
                continue
            end = int(start) + int(size)
            noise[int(start) : end] = rng.standard_normal(int(size))
        return noise


class EggRollNoiserMaterializer:
    """Materialize upstream HyperscaleES noiser perturbations as flat UHD directions."""

    def __init__(
        self,
        runtime: "EggRollJAXRuntime",
        *,
        noiser_name: str,
        rank: int,
        group_size: int,
        freeze_nonlora: bool,
    ) -> None:
        try:
            import optax
            from hyperscalees.noiser import all_noisers
        except ImportError as exc:
            raise ImportError(_DEFAULT_STACK_ERROR) from exc
        if str(noiser_name) not in all_noisers:
            raise ValueError(f"Unknown HyperscaleES noiser {noiser_name!r}. Available: {sorted(all_noisers)}")
        if int(rank) < 1:
            raise ValueError("eggroll_rank must be >= 1.")
        if int(group_size) < 0:
            raise ValueError("eggroll_group_size must be >= 0.")

        self._runtime = runtime
        self._noiser_name = str(noiser_name)
        self._noiser = all_noisers[self._noiser_name]
        self._noiser_module = importlib.import_module(self._noiser.__module__)
        self._frozen_noiser_params, self._noiser_params = self._noiser.init_noiser(
            runtime.policy.params,
            1.0,
            1.0,
            solver=optax.sgd,
            group_size=int(group_size),
            freeze_nonlora=bool(freeze_nonlora),
            rank=int(rank),
            use_batched_update=True,
        )

    def sample(self, x: np.ndarray, *, seed: int) -> np.ndarray:
        rt = self._runtime
        if rt.vector_mode != "absolute":
            raise ValueError("EggRoll noiser perturbations require EggRollJAXRuntime vector_mode='absolute'.")
        x_arr = np.asarray(x, dtype=np.float64)
        params = rt.decode_vector_params(rt.jnp.asarray(x_arr, dtype=rt.jnp.float32))
        iterinfo = (rt.jnp.asarray(0, dtype=rt.jnp.int32), rt.jnp.asarray(2 * int(seed), dtype=rt.jnp.int32))
        noised = rt.jax.tree.map(lambda p, k, m: self._materialize_leaf(p, k, m, iterinfo), params, rt.es_tree_key, rt.policy.es_map)
        noised_x = rt.codec.flatten(noised)
        return np.asarray(noised_x - x_arr, dtype=np.float64)

    def _materialize_leaf(self, param, key, map_class, iterinfo):
        map_id = int(np.asarray(map_class))
        if map_id == 0:
            return self._noiser.get_noisy_standard(self._frozen_noiser_params, self._noiser_params, param, key, iterinfo)
        if map_id == 1:
            return self._materialize_structured_leaf(param, key, iterinfo)
        return param

    def _materialize_structured_leaf(self, param, key, iterinfo):
        if hasattr(self._noiser_module, "get_lora_update_params"):
            return self._materialize_lora_leaf(param, key, iterinfo)
        if hasattr(self._noiser_module, "get_sparse_update_params"):
            return self._materialize_sparse_leaf(param, key, iterinfo)
        raise ValueError(f"No materializer for map_class=1 with noiser {self._noiser_name!r}.")

    def _materialize_lora_leaf(self, param, key, iterinfo):
        rank = int(self._frozen_noiser_params.get("rank", 1))
        base_sigma = self._noiser_params["sigma"] / math.sqrt(float(rank))
        a, b = self._noiser_module.get_lora_update_params(self._frozen_noiser_params, base_sigma, iterinfo, param, key)
        return param + a @ b.T

    def _materialize_sparse_leaf(self, param, key, iterinfo):
        values, idx_a, idx_b = self._noiser_module.get_sparse_update_params(
            self._frozen_noiser_params,
            self._noiser_params["sigma"],
            iterinfo,
            param,
            key,
        )
        return param + self._runtime.jnp.zeros_like(param).at[idx_a, idx_b].add(values)


class EggRollJAXRuntime:
    """Shared flat-vector EggRoll evaluator for UHD and BO-style vector optimizers."""

    def __init__(
        self,
        policy,
        env_conf,
        *,
        steps_per_episode: int = 200,
        eval_episodes: int = 1,
        deterministic_policy: bool = False,
        seed_offset: int = 0,
        embed_num_probes: int = 0,
        vector_mode: str = "absolute",
        param_scale: float = 1.0,
        es_key_fold: int = 31,
        eval_key_fold: int = 32,
        embed_key_fold: int = 33,
        error_cls=ValueError,
        option_label: str = "EggRoll JAX option",
        stack_error_message: str | None = None,
    ) -> None:
        if not hasattr(policy, "model_cls") or not hasattr(policy, "params"):
            raise error_cls("EggRollJAXRuntime requires an EggRoll policy.")
        if env_conf is None:
            raise error_cls("EggRollJAXRuntime requires env_conf.")

        env_name = str(getattr(env_conf, "env_name", ""))
        from problems.eggroll_env_adapters import make_eggroll_env_adapter, supports_eggroll_env_adapter

        if not supports_eggroll_env_adapter(env_name):
            raise error_cls(f"EggRollJAXRuntime requires a supported EggRoll env tag (got {env_name!r}).")

        if vector_mode not in {"absolute", "offset"}:
            raise error_cls("EggRoll JAX option 'vector_mode' must be one of: absolute, offset.")

        self.jax, self.jnp, simple_es_tree_key = require_eggroll_jax_stack(stack_error_message)
        self.policy = policy
        self.env_adapter = make_eggroll_env_adapter(env_name, jax=self.jax, jnp=self.jnp)
        self.steps_per_episode = int(steps_per_episode)
        self.eval_episodes = int(eval_episodes)
        deterministic = as_bool(
            deterministic_policy,
            name="deterministic_policy",
            error_cls=error_cls,
            option_label=option_label,
        )
        self.identity_noiser = IdentityNoiser()
        self._vector_mode = str(vector_mode)
        self._param_scale = float(param_scale)

        if self.steps_per_episode < 1:
            raise error_cls(f"{option_label} 'steps_per_episode' must be >= 1.")
        if self.eval_episodes < 1:
            raise error_cls(f"{option_label} 'eval_episodes' must be >= 1.")
        if int(embed_num_probes) < 0:
            raise error_cls(f"{option_label} 'embed_num_probes' must be >= 0.")
        if self._vector_mode == "offset" and self._param_scale <= 0.0:
            raise error_cls(f"{option_label} 'param_scale' must be > 0.")

        self.codec = EggRollParamCodec(self.jax, self.jnp, policy.params)
        self.dim = self.codec.dim
        self.x0 = self.codec.x0.copy()
        seed = (0 if getattr(policy, "problem_seed", None) is None else int(policy.problem_seed)) + int(seed_offset)
        key = self.jax.random.key(seed & 0xFFFFFFFF)
        es_key = self.jax.random.fold_in(key, int(es_key_fold))
        self.eval_key_base = self.jax.random.fold_in(key, int(eval_key_fold))
        self.embed_key_base = self.jax.random.fold_in(key, int(embed_key_fold))
        self.es_tree_key = simple_es_tree_key(policy.params, es_key, policy.scan_map)
        self.action_selector = EggRollActionSelector(self.jax, self.jnp, deterministic_policy=deterministic)
        self._evaluator = EggRollRuntimeEvaluator(self)
        self._embedder = EggRollRuntimeEmbedder(self)
        self._noise_sampler = EggRollNoiseSampler(self.codec)
        self._noiser_materializers: dict[tuple[str, int, int, bool], EggRollNoiserMaterializer] = {}
        if int(embed_num_probes) > 0:
            self.configure_embedding(int(embed_num_probes))

    @property
    def vector_mode(self) -> str:
        return self._vector_mode

    def decode_vector_params(self, x):
        if self._vector_mode == "offset":
            return self.codec.decode_offset(x, scale=self._param_scale)
        return self.codec.decode_absolute(x)

    def make_policy(self, x: np.ndarray, *, attr_name: str | None = None):
        x_arr = np.asarray(x, dtype=np.float64)
        x_j = self.jnp.asarray(x_arr, dtype=self.jnp.float32)
        policy = self.policy.with_params(self.decode_vector_params(x_j))
        if attr_name is not None:
            setattr(policy, attr_name, x_arr)
        return policy

    def next_eval_keys(self, num_candidates: int):
        return self._evaluator.next_eval_keys(int(num_candidates))

    def evaluate(self, x: np.ndarray, *, seed: int) -> tuple[float, float]:
        return self._evaluator.evaluate(x, seed=int(seed))

    def evaluate_many(self, x_batch: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        return self._evaluator.evaluate_many(x_batch, seed=int(seed))

    def evaluate_many_with_keys(self, x_batch: np.ndarray, keys_batch) -> tuple[np.ndarray, np.ndarray]:
        return self._evaluator.evaluate_many_with_keys(x_batch, keys_batch)

    def evaluate_values_with_keys(self, x_batch: np.ndarray, keys_batch) -> np.ndarray:
        return self._evaluator.evaluate_values_with_keys(x_batch, keys_batch)

    def configure_embedding(self, num_probes: int) -> None:
        self._embedder.configure(int(num_probes))

    def embed_many(self, x_batch: np.ndarray) -> np.ndarray:
        return self._embedder.embed_many(x_batch)

    def embed(self, x: np.ndarray) -> np.ndarray:
        return self._embedder.embed(x)

    def sample_noise(
        self,
        *,
        seed: int,
        num_dim_target: float | None = None,
        num_module_target: float | None = None,
    ) -> np.ndarray:
        return self._noise_sampler.sample(
            seed=int(seed),
            num_dim_target=num_dim_target,
            num_module_target=num_module_target,
        )

    def sample_eggroll_noiser_noise(
        self,
        x: np.ndarray,
        *,
        seed: int,
        noiser_name: str = "eggroll",
        rank: int = 1,
        group_size: int = 0,
        freeze_nonlora: bool = False,
    ) -> np.ndarray:
        key = (str(noiser_name), int(rank), int(group_size), bool(freeze_nonlora))
        materializer = self._noiser_materializers.get(key)
        if materializer is None:
            materializer = EggRollNoiserMaterializer(
                self,
                noiser_name=key[0],
                rank=key[1],
                group_size=key[2],
                freeze_nonlora=key[3],
            )
            self._noiser_materializers[key] = materializer
        return materializer.sample(x, seed=int(seed))
