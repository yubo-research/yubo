from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


def _require_hyperscalees():
    try:
        import jax
        import jax.numpy as jnp
        from hyperscalees.models.rl import ActorCriticMLP
    except ImportError as exc:
        raise ImportError(
            "EggRoll policy tags require the separate HyperscaleES environment. "
            "Run admin/setup-hyperscalees.sh first, then use the plain python CLI from that environment."
        ) from exc
    return jax, jnp, ActorCriticMLP


@dataclass(frozen=True)
class EggRollActorCriticMLPSpec:
    hidden_dim: int
    layers: int
    activation: str = "silu"
    use_bias: bool = True
    have_critic: bool = False
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if int(self.hidden_dim) < 1:
            raise ValueError("EggRoll hidden_dim must be >= 1.")
        if int(self.layers) < 1:
            raise ValueError("EggRoll layers must be >= 1.")
        if self.activation not in {"relu", "silu", "pqn", "tanh"}:
            raise ValueError("EggRoll activation must be one of: relu, silu, pqn, tanh.")
        if self.dtype != "float32":
            raise ValueError("Only float32 EggRoll policies are currently supported.")

    @property
    def hyperscalees_activation(self) -> str:
        # Upstream HyperscaleES ActorCriticMLP exposes relu/silu/pqn. The paper
        # MARL configs use tanh tags, so map them onto the closest smooth option
        # until a tanh activation lands upstream.
        if self.activation == "tanh":
            return "silu"
        return self.activation


class EggRollActorCriticMLPPolicy:
    """HyperscaleES ActorCriticMLP policy metadata for the EggRoll designer.

    This policy is intentionally not evaluated by the normal Python trajectory
    collector. The EggRoll designer owns JAX/Gymnax rollout, noising, and update.
    """

    def __init__(self, env_runtime: Any, spec: EggRollActorCriticMLPSpec) -> None:
        jax, jnp, actor_critic_mlp = _require_hyperscalees()
        env_name = str(getattr(env_runtime, "env_name", ""))
        from problems.eggroll_env_adapters import supports_eggroll_env_adapter

        if not supports_eggroll_env_adapter(env_name):
            raise ValueError(f"EggRoll policies require a supported EggRoll adapter env tag (got env_name={env_name!r}).")

        self.problem_seed = getattr(env_runtime, "problem_seed", None)
        self.env_name = env_name
        self.spec = spec
        self.model_cls = actor_critic_mlp

        obs_space = getattr(env_runtime, "state_space", None)
        act_space = getattr(env_runtime, "action_space", None)
        if obs_space is None or act_space is None:
            raise ValueError("EggRoll policy construction requires resolved Gymnax observation/action spaces.")
        self.obs_space = obs_space
        self.act_space = act_space

        seed = 0 if self.problem_seed is None else int(self.problem_seed) & 0xFFFFFFFF
        key = jax.random.key(seed)
        dtype = jnp.float32
        init = actor_critic_mlp.rand_init(
            key,
            n_embd=int(spec.hidden_dim),
            obs_space=obs_space,
            act_space=act_space,
            n_layers=int(spec.layers),
            use_bias=bool(spec.use_bias),
            activation=str(spec.hyperscalees_activation),
            have_critic=bool(spec.have_critic),
            dtype=dtype,
        )
        self.frozen_params, self.params, self.scan_map, self.es_map = init

    def num_params(self) -> int:
        jax, _jnp, _actor_critic_mlp = _require_hyperscalees()
        return int(sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(self.params)))

    def clone(self):
        return copy.deepcopy(self)

    def with_params(self, params):
        policy = self.clone()
        policy.params = params
        return policy

    def get_params(self):
        raise NotImplementedError("EggRollPolicy params are managed by EggRollDesigner, not generic BO designers.")

    def set_params(self, _flat_params) -> None:
        raise NotImplementedError("EggRollPolicy params are managed by EggRollDesigner, not generic BO designers.")

    def __call__(self, _state):
        raise NotImplementedError("EggRollPolicy must be evaluated with optimizer.name='eggroll'.")


class EggRollActorCriticMLPPolicyFactory:
    def __init__(self, spec: EggRollActorCriticMLPSpec) -> None:
        self._spec = spec

    def __call__(self, env_runtime: Any) -> EggRollActorCriticMLPPolicy:
        from problems.isaaclab_env_adapters import is_isaaclab_env_tag

        if is_isaaclab_env_tag(str(getattr(env_runtime, "env_name", ""))):
            from policies.mlp_policy import MLPPolicyFactory

            hidden_sizes = tuple(int(self._spec.hidden_dim) for _ in range(int(self._spec.layers)))
            return MLPPolicyFactory(hidden_sizes)(env_runtime)
        return EggRollActorCriticMLPPolicy(env_runtime, self._spec)
