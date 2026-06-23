from __future__ import annotations

from typing import Any, TypeAlias

from problems import jax_env_core as core
from problems.jaxtyping_compat import Array, Float, PRNGKeyArray
from problems.mujoco_gl import normalize_mujoco_gl_for_platform

Obs: TypeAlias = Any
State: TypeAlias = Any
Action: TypeAlias = Any


def _mjx_impl() -> str:
    normalize_mujoco_gl_for_platform()
    try:
        import mujoco_warp  # noqa: F401
        import warp
    except Exception:
        return "jax"
    if not hasattr(getattr(warp, "types", None), "warp_type_to_np_dtype"):
        return "jax"
    return "warp"


def _action_space(env, spaces, jnp):
    model = getattr(env, "_mj_model", None)
    action_dim = int(getattr(model, "nu", 0))
    if action_dim <= 0:
        raise ValueError("MuJoCo Playground env does not expose a positive actuator count.")
    shape = (action_dim,)
    return spaces.Box(
        low=-jnp.ones(shape, dtype=jnp.float32),
        high=jnp.ones(shape, dtype=jnp.float32),
        shape=shape,
        dtype=jnp.float32,
    )


def _policy_obs(obs, jax, jnp):
    if isinstance(obs, dict) and "state" in obs:
        return jnp.asarray(obs["state"], dtype=jnp.float32)
    return core._flat_obs(obs, jax, jnp)


class MujocoPlaygroundAdapter:
    def __init__(self, env_name: str, *, jax, jnp) -> None:
        raw_name = env_name.split(":", 1)[1]

        from gymnax.environments import spaces

        normalize_mujoco_gl_for_platform()
        from mujoco_playground import registry

        self._jax = jax
        self._jnp = jnp
        env = registry.load(raw_name, config_overrides={"impl": _mjx_impl()})
        self.env = env
        sample_state = env.reset(jax.random.key(0))
        sample_obs = _policy_obs(sample_state.obs, jax, jnp)
        self.observation_space = core._gymnax_box_from_shape(
            spaces,
            jnp,
            tuple(int(v) for v in sample_obs.shape),
        )
        self.action_space = _action_space(env, spaces, jnp)

    def reset(self, key: PRNGKeyArray) -> tuple[Obs, State]:
        state = self.env.reset(key)
        return _policy_obs(state.obs, self._jax, self._jnp), state

    def step(self, _key: PRNGKeyArray, state: State, action: Action) -> core.JaxStepResult:
        _step_key, reset_key = self._jax.random.split(_key)
        action = self.clip_action(action)
        next_state = self.env.step(state, action)
        done_bool = self._jnp.asarray(next_state.done, dtype=bool)
        out_state = self._jax.lax.cond(
            done_bool,
            lambda _: self.env.reset(reset_key),
            lambda _: next_state,
            operand=None,
        )
        obs = _policy_obs(out_state.obs, self._jax, self._jnp)
        reward = self._jnp.asarray(next_state.reward, dtype=self._jnp.float32)
        terminated = self._jnp.asarray(getattr(next_state, "terminated", next_state.done), dtype=self._jnp.float32)
        truncated = self._jnp.asarray(
            getattr(next_state, "truncated", self._jnp.zeros_like(next_state.done)),
            dtype=self._jnp.float32,
        )
        return core.JaxStepResult(
            obs=obs,
            state=out_state,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=next_state.metrics,
        )

    def clip_action(self, action: Float[Array, "..."]) -> Float[Array, "..."]:
        return core._clip_box_action(self.action_space, self._jnp, action)
