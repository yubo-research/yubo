from __future__ import annotations

from typing import Any

import numpy as np

import optimizer.eggroll_designer_iter as jax_iter_helpers
import optimizer.eggroll_designer_jit as jit_helpers
import optimizer.eggroll_designer_nanoegg as nanoegg_helpers
from optimizer.designer_errors import NoSuchDesignerError
from optimizer.eggroll_designer_config import _EggRollDesignerConfig
from optimizer.eggroll_designer_types import (
    EggRollState,
    _NoiserBundle,
    _SeedState,
)
from optimizer.eggroll_options import eggroll_bool as _as_bool
from optimizer.eggroll_options import unit_decay as _as_unit_decay
from optimizer.eggroll_runtime_types import _EggRollStack
from optimizer.optimizer_types import IterateResult


def _require_stack():
    try:
        import jax
        import jax.numpy as jnp
        import optax
        from hyperscalees.models.common import simple_es_tree_key
        from hyperscalees.noiser import all_noisers
    except ImportError as exc:
        raise ImportError(
            "EggRollDesigner requires the separate HyperscaleES environment. Run the Pixi setup task first, then use that Pixi environment."
        ) from exc
    return _EggRollStack(
        jax=jax,
        jnp=jnp,
        optax=optax,
        simple_es_tree_key=simple_es_tree_key,
        all_noisers=all_noisers,
    )


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


def _scheduled_lr(jnp, lr: float, lr_decay: float):
    return _learning_rate_schedule(jnp, base_lr=lr, lr_decay=lr_decay) if lr_decay != 1.0 else lr


def _seed_state(jax, policy, seed_offset: int) -> _SeedState:
    seed = (0 if getattr(policy, "problem_seed", None) is None else int(policy.problem_seed)) + int(seed_offset)
    key = jax.random.key(seed & 0xFFFFFFFF)
    return _SeedState(
        es_key=jax.random.fold_in(key, 1),
        train_key=jax.random.fold_in(key, 2),
        eval_key=jax.random.fold_in(key, 3),
    )


def _attach_policy_runtime_metadata(policy, cfg: _EggRollDesignerConfig, *, steps_per_episode: int, num_envs: int) -> None:
    policy._eggroll_steps_per_episode = int(steps_per_episode)
    policy._eggroll_num_envs = int(num_envs)
    policy._eggroll_deterministic_policy = _as_bool(cfg.deterministic_policy, name="deterministic_policy")


def _validate_positive_jax_options(sigma: float, lr: float, steps: int, num_envs: int, rank: int) -> None:
    if sigma <= 0.0:
        raise NoSuchDesignerError("EggRoll option 'sigma' must be > 0.")
    if lr <= 0.0:
        raise NoSuchDesignerError("EggRoll option 'lr' must be > 0.")
    if steps < 1:
        raise NoSuchDesignerError("EggRoll option 'steps' must be >= 1.")
    if num_envs < 1:
        raise NoSuchDesignerError("EggRoll option 'num_envs' must be >= 1.")
    if rank < 1:
        raise NoSuchDesignerError("EggRoll option 'rank' must be >= 1.")


def _load_nanoegg_optax():
    try:
        import optax as optax_mod
    except ImportError as exc:
        raise ImportError("NanoEgg EggRoll requires optax. Run the Pixi setup task first.") from exc
    return optax_mod


def _validate_nanoegg_options(designer, cfg: _EggRollDesignerConfig) -> None:
    _validate_positive_jax_options(
        designer._sigma,
        float(cfg.lr),
        designer._steps_per_episode,
        designer._num_envs,
        designer._rank,
    )
    if designer._nanoegg_batch_size < 1:
        raise NoSuchDesignerError("NanoEgg EggRoll option 'batch_size' must be >= 1.")
    if int(cfg.search_dim) < 1:
        raise NoSuchDesignerError("NanoEgg EggRoll option 'search_dim' must be >= 1.")
    if float(cfg.delta_scale) <= 0.0:
        raise NoSuchDesignerError("NanoEgg EggRoll option 'delta_scale' must be > 0.")


class EggRollDesigner:
    """JAX HyperscaleES optimizer integrated into the experiment runner."""

    def __init__(
        self,
        policy,
        env_conf,
        **options,
    ) -> None:
        cfg = _designer_config(options)
        self._state = EggRollState()
        if bool(getattr(policy, "is_nanoegg_pretrain_policy", False)):
            self._init_nanoegg(policy, cfg)
            return
        if _should_use_external_scoring(env_conf, cfg):
            from optimizer.eggroll_external import init_external

            init_external(self, policy, env_conf, cfg)
            return
        self._init_jax(policy, env_conf, cfg)

    def _init_jax(self, policy, env_conf, cfg: _EggRollDesignerConfig) -> None:
        if not hasattr(policy, "model_cls") or not hasattr(policy, "params"):
            raise NoSuchDesignerError("Designer 'eggroll' requires an EggRoll policy tag, e.g. policy_tag='eggroll-ac-mlp-64x2-silu'.")
        if env_conf is None:
            raise NoSuchDesignerError("Designer 'eggroll' requires env_conf.")
        env_name = str(getattr(env_conf, "env_name", ""))
        stack = _require_stack()
        self._validate_eggroll_env(env_name)
        self._assign_jax_state(policy, stack, cfg)
        seed_state = _seed_state(stack.jax, policy, int(cfg.seed_offset))
        self._state.train_key = seed_state.train_key
        self._state.eval_key = seed_state.eval_key
        setattr(env_conf, "eggroll_eval_envs", int(cfg.num_envs))
        env_adapter = self._make_env_adapter(env_name, env_conf)
        noiser_bundle = self._init_jax_noiser(stack, cfg)
        es_tree_key = stack.simple_es_tree_key(self._state.params, seed_state.es_key, policy.scan_map)
        self._evaluate_population, self._evaluate_policy, self._update_params = jit_helpers.build_jitted_fns(
            self,
            jit_helpers.JittedFnConfig(
                env_adapter=env_adapter,
                noiser=noiser_bundle.noiser,
                frozen_noiser_params=noiser_bundle.frozen_params,
                frozen_params=policy.frozen_params,
                es_tree_key=es_tree_key,
                es_map=policy.es_map,
                rank_transform=_as_bool(cfg.rank_transform, name="rank_transform"),
                deterministic_policy=_as_bool(cfg.deterministic_policy, name="deterministic_policy"),
            ),
        )

    def _validate_eggroll_env(self, env_name: str) -> None:
        from optimizer.eggroll_env_validation import validate_eggroll_jax_objective_env

        validate_eggroll_jax_objective_env(env_name, NoSuchDesignerError)

    def _assign_jax_state(self, policy, stack: _EggRollStack, cfg: _EggRollDesignerConfig) -> None:
        self._policy = policy
        self._jax = stack.jax
        self._jnp = stack.jnp
        self._steps_per_episode = int(cfg.steps)
        self._num_envs = int(cfg.num_envs)
        self._sigma = float(cfg.sigma)
        self._sigma_decay = _as_unit_decay(cfg.sigma_decay, name="sigma_decay")
        self._suppress_noiser_stdout = _as_bool(cfg.suppress_noiser_stdout, name="suppress_noiser_stdout")
        _attach_policy_runtime_metadata(
            policy,
            cfg,
            steps_per_episode=self._steps_per_episode,
            num_envs=self._num_envs,
        )
        self._state.best_datum = None
        self._state.epoch = 0
        self._state.params = policy.params
        _validate_positive_jax_options(
            self._sigma,
            float(cfg.lr),
            self._steps_per_episode,
            self._num_envs,
            int(cfg.rank),
        )

    def _make_env_adapter(self, env_name: str, env_conf: Any):
        from problems.jax_env_factory import make_jax_env_adapter

        return make_jax_env_adapter(env_name, jax=self._jax, jnp=self._jnp, env_runtime=env_conf)

    def _init_jax_noiser(self, stack: _EggRollStack, cfg: _EggRollDesignerConfig) -> _NoiserBundle:
        if cfg.noiser not in stack.all_noisers:
            raise NoSuchDesignerError(f"Unknown HyperscaleES noiser '{cfg.noiser}'. Available: {sorted(stack.all_noisers)}")
        solver = _solver(
            stack.optax,
            str(cfg.optax),
            b1=float(cfg.b1),
            b2=float(cfg.b2),
            weight_decay=float(cfg.weight_decay),
        )
        noiser = stack.all_noisers[cfg.noiser]
        scheduled_lr = _scheduled_lr(self._jnp, float(cfg.lr), _as_unit_decay(cfg.lr_decay, name="lr_decay"))
        frozen_noiser_params, noiser_params = noiser.init_noiser(
            self._state.params,
            self._sigma,
            scheduled_lr,
            solver=solver,
            solver_kwargs=_solver_kwargs(
                str(cfg.optax),
                b1=float(cfg.b1),
                b2=float(cfg.b2),
                weight_decay=float(cfg.weight_decay),
            ),
            group_size=int(cfg.group_size),
            freeze_nonlora=_as_bool(cfg.freeze_nonlora, name="freeze_nonlora"),
            noise_reuse=int(cfg.noise_reuse),
            rank=int(cfg.rank),
            use_batched_update=_as_bool(cfg.use_batched_update, name="use_batched_update"),
        )
        self._state.noiser_params = noiser_params
        return _NoiserBundle(noiser=noiser, frozen_params=frozen_noiser_params, params=noiser_params)

    def _init_nanoegg(self, policy, cfg: _EggRollDesignerConfig) -> None:
        if cfg.noiser != "eggroll":
            raise NoSuchDesignerError("NanoEgg pretraining currently supports EggRoll option noiser='eggroll' only.")
        optax_mod = _load_nanoegg_optax()
        solver = _solver(
            optax_mod,
            str(cfg.optax),
            b1=float(cfg.b1),
            b2=float(cfg.b2),
            weight_decay=float(cfg.weight_decay),
        )
        self._assign_nanoegg_state(policy, cfg)
        self._objective = self._build_nanoegg_objective(policy, cfg)
        self._state.x = np.asarray(self._objective.x0, dtype=np.float64).copy()
        self._init_nanoegg_optimizer(optax_mod, solver, cfg)

    def _assign_nanoegg_state(self, policy, cfg: _EggRollDesignerConfig) -> None:
        self._policy = policy
        self._is_nanoegg = True
        self._sigma = float(cfg.sigma)
        self._sigma_decay = _as_unit_decay(cfg.sigma_decay, name="sigma_decay")
        self._rank = int(cfg.rank)
        self._rank_transform = _as_bool(cfg.rank_transform, name="rank_transform")
        self._group_size = int(cfg.group_size)
        self._freeze_nonlora = _as_bool(cfg.freeze_nonlora, name="freeze_nonlora")
        self._noise_reuse = int(cfg.noise_reuse)
        self._nanoegg_batch_size = int(cfg.batch_size)
        self._steps_per_episode = int(cfg.steps)
        self._num_envs = int(cfg.num_envs)
        self._state.best_datum = None
        self._state.epoch = 0
        _validate_nanoegg_options(self, cfg)

    def _build_nanoegg_objective(self, policy, cfg: _EggRollDesignerConfig):
        self._nanoegg_log(
            f"building objective env={getattr(policy, 'env_name', 'unknown')} policy={getattr(policy, 'policy_tag', 'unknown')} "
            f"search_dim={int(cfg.search_dim)}"
        )
        return policy.make_objective(
            search_dim=int(cfg.search_dim),
            delta_scale=float(cfg.delta_scale),
            generation_length=None if cfg.generation_length is None else int(cfg.generation_length),
            num_envs=self._num_envs,
            lora_only=_as_bool(cfg.lora_only, name="lora_only"),
            basis_max_leaves=None if cfg.basis_max_leaves is None else int(cfg.basis_max_leaves),
            sub_dataset_size=None if cfg.sub_dataset_size is None else int(cfg.sub_dataset_size),
            hf_home=cfg.hf_home,
        )

    def _nanoegg_log(self, message: str) -> None:
        print(f"NANOEGG_EGGROLL: {message}", flush=True)

    def _init_nanoegg_optimizer(self, optax_mod, solver, cfg: _EggRollDesignerConfig) -> None:
        scheduled_lr = _scheduled_lr(
            self._objective.jnp,
            float(cfg.lr),
            _as_unit_decay(cfg.lr_decay, name="lr_decay"),
        )
        self._nanoegg_tx = solver(
            scheduled_lr,
            **_solver_kwargs(
                str(cfg.optax),
                b1=float(cfg.b1),
                b2=float(cfg.b2),
                weight_decay=float(cfg.weight_decay),
            ),
        )
        self._nanoegg_apply_updates = optax_mod.apply_updates
        self._state.opt_state = self._nanoegg_tx.init(self._objective.jnp.asarray(self._state.x, dtype=self._objective.jnp.float32))

    def best_datum(self):
        return self._state.best_datum

    def _current_sigma(self) -> float:
        return self._sigma * (self._sigma_decay**self._state.epoch)

    def _iterate_nanoegg(self, _data, num_arms: int, *, telemetry=None) -> IterateResult:
        return nanoegg_helpers.iterate_nanoegg(self, self._state, _data, num_arms, telemetry=telemetry)

    def iterate(self, _data, num_arms: int, *, telemetry=None) -> IterateResult:
        if bool(getattr(self, "_is_external", False)):
            from optimizer.eggroll_external import iterate_external

            return iterate_external(self, self._state, _data, num_arms, telemetry=telemetry)
        if bool(getattr(self, "_is_nanoegg", False)):
            return self._iterate_nanoegg(_data, num_arms, telemetry=telemetry)

        return jax_iter_helpers.iterate_jax(self, self._state, _data, num_arms, telemetry=telemetry)

    def stop(self):
        objective = getattr(self, "_objective", None)
        if objective is not None and hasattr(objective, "close"):
            objective.close()


def _should_use_external_scoring(env_conf, cfg: _EggRollDesignerConfig) -> bool:
    if env_conf is None:
        return False
    if bool(cfg.jax_sim):
        return False
    from optimizer.eggroll_external import supports_external_scoring_env

    return supports_external_scoring_env(env_conf)
