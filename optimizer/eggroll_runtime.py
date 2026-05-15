from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from optimizer.eggroll_runtime_core import (
    EggRollActionSelector,
    EggRollParamCodec,
    IdentityNoiser,
    as_bool,
    require_eggroll_jax_stack,
)
from optimizer.eggroll_runtime_embed import EggRollRuntimeEmbedder
from optimizer.eggroll_runtime_eval import EggRollRuntimeEvaluator
from optimizer.eggroll_runtime_noise import (
    EggRollNoiserMaterializer,
    EggRollNoiseSampler,
)


@dataclass(frozen=True)
class EggRollRuntimeConfig:
    steps_per_episode: int = 200
    num_envs: int = 1
    deterministic_policy: bool = False
    seed_offset: int = 0
    embed_num_probes: int = 0
    vector_mode: str = "absolute"
    param_scale: float = 1.0
    es_key_fold: int = 31
    eval_key_fold: int = 32
    embed_key_fold: int = 33
    error_cls: Any = ValueError
    option_label: str = "EggRoll JAX option"
    stack_error_message: str | None = None

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> EggRollRuntimeConfig:
        return cls(**kwargs)


class EggRollJAXRuntime:
    """Shared flat-vector EggRoll evaluator for UHD and BO-style vector optimizers."""

    def __init__(
        self,
        policy,
        env_conf,
        config: EggRollRuntimeConfig | None = None,
        **kwargs: Any,
    ) -> None:
        cfg = config if config is not None else EggRollRuntimeConfig.from_kwargs(**kwargs)
        _validate_policy_env(policy, env_conf, cfg)
        self._assign_core(policy, env_conf, cfg)
        self._init_seeds_and_codec(policy, cfg)
        self._init_helpers(cfg)

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

    def _assign_core(self, policy, env_conf, cfg: EggRollRuntimeConfig) -> None:
        env_name = str(getattr(env_conf, "env_name", ""))
        from problems.jax_env_factory import make_jax_env_adapter

        self.jax, self.jnp, self._simple_es_tree_key = require_eggroll_jax_stack(cfg.stack_error_message)
        self.policy = policy
        self.env_adapter = make_jax_env_adapter(env_name, jax=self.jax, jnp=self.jnp)
        self.steps_per_episode = int(cfg.steps_per_episode)
        self.num_envs = int(cfg.num_envs)
        self.identity_noiser = IdentityNoiser()
        self._vector_mode = str(cfg.vector_mode)
        self._param_scale = float(cfg.param_scale)
        self.action_selector = EggRollActionSelector(self.jax, self.jnp, deterministic_policy=_deterministic(cfg))

    def _init_seeds_and_codec(self, policy, cfg: EggRollRuntimeConfig) -> None:
        self.codec = EggRollParamCodec(self.jax, self.jnp, policy.params)
        self.dim = self.codec.dim
        self.x0 = self.codec.x0.copy()
        seed = (0 if getattr(policy, "problem_seed", None) is None else int(policy.problem_seed)) + int(cfg.seed_offset)
        key = self.jax.random.key(seed & 0xFFFFFFFF)
        es_key = self.jax.random.fold_in(key, int(cfg.es_key_fold))
        self.eval_key_base = self.jax.random.fold_in(key, int(cfg.eval_key_fold))
        self.embed_key_base = self.jax.random.fold_in(key, int(cfg.embed_key_fold))
        self.es_tree_key = self._simple_es_tree_key(policy.params, es_key, policy.scan_map)

    def _init_helpers(self, cfg: EggRollRuntimeConfig) -> None:
        self._evaluator = EggRollRuntimeEvaluator(self)
        self._embedder = EggRollRuntimeEmbedder(self)
        self._noise_sampler = EggRollNoiseSampler(self.codec)
        self._noiser_materializers: dict[tuple[str, int, int, bool], EggRollNoiserMaterializer] = {}
        if int(cfg.embed_num_probes) > 0:
            self.configure_embedding(int(cfg.embed_num_probes))


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


EggRollJAXRuntime.sample_eggroll_noiser_noise = sample_eggroll_noiser_noise


def _deterministic(cfg: EggRollRuntimeConfig) -> bool:
    return as_bool(
        cfg.deterministic_policy,
        name="deterministic_policy",
        error_cls=cfg.error_cls,
        option_label=cfg.option_label,
    )


def _validate_policy_env(policy, env_conf, cfg: EggRollRuntimeConfig) -> None:
    if not hasattr(policy, "model_cls") or not hasattr(policy, "params"):
        raise cfg.error_cls("EggRollJAXRuntime requires an EggRoll policy.")
    if env_conf is None:
        raise cfg.error_cls("EggRollJAXRuntime requires env_conf.")
    _validate_env_name(str(getattr(env_conf, "env_name", "")), cfg)
    _validate_config_values(cfg)


def _validate_env_name(env_name: str, cfg: EggRollRuntimeConfig) -> None:
    from optimizer.eggroll_env_validation import validate_eggroll_jax_objective_env

    validate_eggroll_jax_objective_env(env_name, cfg.error_cls)


def _validate_config_values(cfg: EggRollRuntimeConfig) -> None:
    if cfg.vector_mode not in {"absolute", "offset"}:
        raise cfg.error_cls("EggRoll JAX option 'vector_mode' must be one of: absolute, offset.")
    if int(cfg.steps_per_episode) < 1:
        raise cfg.error_cls(f"{cfg.option_label} 'steps_per_episode' must be >= 1.")
    if int(cfg.num_envs) < 1:
        raise cfg.error_cls(f"{cfg.option_label} 'num_envs' must be >= 1.")
    if int(cfg.embed_num_probes) < 0:
        raise cfg.error_cls(f"{cfg.option_label} 'embed_num_probes' must be >= 0.")
    if cfg.vector_mode == "offset" and float(cfg.param_scale) <= 0.0:
        raise cfg.error_cls(f"{cfg.option_label} 'param_scale' must be > 0.")


__all__ = [
    "EggRollActionSelector",
    "EggRollJAXRuntime",
    "EggRollNoiseSampler",
    "EggRollNoiserMaterializer",
    "EggRollParamCodec",
    "EggRollRuntimeConfig",
    "EggRollRuntimeEmbedder",
    "EggRollRuntimeEvaluator",
    "IdentityNoiser",
    "as_bool",
    "require_eggroll_jax_stack",
]
