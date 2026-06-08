from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from problems.isaaclab_batch_host import host_reset_batch, host_step_batch
from problems.isaaclab_env_adapters import is_isaaclab_env_tag, make_isaaclab_env
from problems.isaaclab_jax_spaces import gymnax_spaces_from_host


def _per_env_obs_dim(host: Any, num_envs: int) -> int:
    space = getattr(host, "observation_space", None)
    shape = tuple(int(v) for v in getattr(space, "shape", ()))
    if not shape:
        return 1
    if len(shape) == 1:
        flat = int(shape[0])
        n = int(num_envs)
        if n > 1 and flat % n == 0 and flat != n:
            return flat // n
        return flat
    if shape[0] == int(num_envs):
        return int(np.prod(shape[1:]))
    total = int(np.prod(shape))
    return max(1, total // max(1, int(num_envs)))


class IsaacLabJaxVectorState(NamedTuple):
    episode_step: Any


class IsaacLabJaxVectorAdapter:
    """Batched Isaac Lab sim for JAX: one step_batch per scan iteration."""

    vector_num_envs: int

    def __init__(
        self,
        env_name: str,
        *,
        jax,
        jnp,
        num_envs: int,
        problem_seed: int = 0,
        device: str | None = None,
        max_steps: int = 1000,
        headless: bool = True,
        launcher_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not is_isaaclab_env_tag(str(env_name)):
            raise ValueError(f"IsaacLabJaxVectorAdapter requires an isaaclab: env tag (got {env_name!r}).")
        self.vector_num_envs = max(1, int(num_envs))
        self._jax = jax
        self._jnp = jnp
        self._env_name = str(env_name)
        self._problem_seed = int(problem_seed)
        self._device = device
        self._max_steps = max(1, int(max_steps))
        self._headless = bool(headless)
        self._launcher_kwargs = launcher_kwargs
        self._host = None
        self._obs_dim = 0
        self._reset_batch_fn = None
        self._step_batch_fn = None
        self._ensure_host()
        self._obs_dim = _per_env_obs_dim(self._host, self.vector_num_envs)
        self.observation_space, self.action_space = gymnax_spaces_from_host(self._host, jax=jax, jnp=jnp)
        if int(self.observation_space.shape[0]) != self._obs_dim:
            from gymnax.environments import spaces

            self.observation_space = spaces.Box(
                low=jnp.full((self._obs_dim,), -jnp.inf, dtype=jnp.float32),
                high=jnp.full((self._obs_dim,), jnp.inf, dtype=jnp.float32),
                shape=(self._obs_dim,),
                dtype=jnp.float32,
            )
        self._reset_batch_fn = _make_reset_batch_fn(self._host)
        self._step_batch_fn = _make_step_batch_fn(self._host, self._max_steps)

    def close(self) -> None:
        if self._host is not None:
            self._host.close()
            self._host = None

    def reset(self, key: Any) -> tuple[Any, IsaacLabJaxVectorState]:
        seed = self._jax.random.randint(key, (), 0, 2**31 - 1, dtype=self._jnp.int32)
        obs = self._reset_batch_obs(seed)
        steps = self._jnp.zeros((self.vector_num_envs,), dtype=self._jnp.int32)
        return obs, IsaacLabJaxVectorState(episode_step=steps)

    def step_batched(
        self,
        state: IsaacLabJaxVectorState,
        actions: Any,
    ) -> tuple[Any, IsaacLabJaxVectorState, Any, Any, Any]:
        obs, reward, term, trunc, next_step = self._step_batch_obs(actions, state.episode_step)
        return obs, IsaacLabJaxVectorState(episode_step=next_step), reward, term, trunc

    def clip_action(self, action: Any) -> Any:
        return self._jnp.clip(
            self._jnp.asarray(action, dtype=self._jnp.float32),
            self.action_space.low,
            self.action_space.high,
        )

    def _ensure_host(self) -> None:
        if self._host is not None:
            return
        self._host = make_isaaclab_env(
            self._env_name,
            headless=self._headless,
            num_envs=self.vector_num_envs,
            batched=True,
            device=self._device,
            launcher_kwargs=self._launcher_kwargs,
            seed=self._problem_seed,
        )

    def _reshape_batch_obs(self, obs: Any) -> Any:
        n = self.vector_num_envs
        obs = self._jnp.asarray(obs, dtype=self._jnp.float32)
        return obs.reshape((n, self._obs_dim))

    def _reset_batch_obs(self, seed: Any) -> Any:
        n = self.vector_num_envs
        if getattr(self._jnp, "__name__", "") == "numpy":
            return self._reshape_batch_obs(self._reset_batch_fn(int(seed)))
        obs = self._jax.pure_callback(
            self._reset_batch_fn,
            self._jax.ShapeDtypeStruct((n, self._obs_dim), self._jnp.float32),
            self._jnp.asarray(seed, dtype=self._jnp.int32),
            vmap_method="sequential",
        )
        return self._reshape_batch_obs(obs)

    def _step_batch_obs(self, actions: Any, episode_step: Any) -> tuple[Any, Any, Any, Any, Any]:
        n = self.vector_num_envs
        actions_f32 = self._jnp.asarray(actions, dtype=self._jnp.float32)
        steps_i32 = self._jnp.asarray(episode_step, dtype=self._jnp.int32)
        if getattr(self._jnp, "__name__", "") == "numpy":
            obs, reward, term, trunc, next_step = self._step_batch_fn(
                np.asarray(actions_f32, dtype=np.float32),
                np.asarray(steps_i32, dtype=np.int32),
            )
            return (
                self._reshape_batch_obs(obs),
                self._jnp.asarray(reward, dtype=self._jnp.float32),
                self._jnp.asarray(term, dtype=self._jnp.float32),
                self._jnp.asarray(trunc, dtype=self._jnp.float32),
                self._jnp.asarray(next_step, dtype=self._jnp.int32),
            )
        out_struct = (
            self._jax.ShapeDtypeStruct((n, self._obs_dim), self._jnp.float32),
            self._jax.ShapeDtypeStruct((n,), self._jnp.float32),
            self._jax.ShapeDtypeStruct((n,), self._jnp.float32),
            self._jax.ShapeDtypeStruct((n,), self._jnp.float32),
            self._jax.ShapeDtypeStruct((n,), self._jnp.int32),
        )
        obs, reward, term, trunc, next_step = self._jax.pure_callback(
            self._step_batch_fn,
            out_struct,
            actions_f32,
            steps_i32,
            vmap_method="sequential",
        )
        return self._reshape_batch_obs(obs), reward, term, trunc, next_step


def _make_reset_batch_fn(host: Any):
    def _reset_batch(seed: int) -> np.ndarray:
        return host_reset_batch(host, int(seed))

    return _reset_batch


def _make_step_batch_fn(host: Any, max_steps: int):
    def _step_batch(actions: np.ndarray, episode_steps: np.ndarray) -> tuple[np.ndarray, ...]:
        return host_step_batch(host, actions, episode_steps, max_steps=int(max_steps))

    return _step_batch
