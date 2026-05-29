from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from problems import jax_env_core as core
from problems.isaaclab_env_adapters import is_isaaclab_env_tag, make_isaaclab_env
from problems.isaaclab_jax_spaces import gymnax_spaces_from_host


class IsaacLabJaxState(NamedTuple):
    episode_step: Any


def resolve_isaaclab_jax_num_envs(env_runtime: Any | None) -> int:
    if env_runtime is None:
        return 1
    pop = int(getattr(env_runtime, "eggroll_population", 0) or 0)
    eval_n = int(getattr(env_runtime, "eggroll_eval_envs", 1) or 1)
    return max(1, pop, eval_n)


def make_isaaclab_jax_adapter(
    env_name: str,
    *,
    jax,
    jnp,
    env_runtime: Any | None = None,
    problem_seed: int = 0,
    device: str | None = None,
    max_steps: int = 1000,
    headless: bool = True,
    launcher_kwargs: dict[str, Any] | None = None,
):
    num_envs = resolve_isaaclab_jax_num_envs(env_runtime)
    common = dict(
        env_name=env_name,
        jax=jax,
        jnp=jnp,
        problem_seed=problem_seed,
        device=device,
        max_steps=max_steps,
        headless=headless,
        launcher_kwargs=launcher_kwargs,
    )
    if num_envs > 1:
        from problems.isaaclab_jax_vector_env import IsaacLabJaxVectorAdapter

        return IsaacLabJaxVectorAdapter(**common, num_envs=num_envs)
    return IsaacLabJaxAdapter(**common)


def _host_reset(host: Any, seed: int) -> np.ndarray:
    obs, _info = host.reset(seed=int(seed))
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _host_step(host: Any, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool]:
    obs, reward, terminated, truncated, _info = host.step(np.asarray(action, dtype=np.float32))
    return (
        np.asarray(obs, dtype=np.float32).reshape(-1),
        float(reward),
        bool(terminated),
        bool(truncated),
    )


class IsaacLabJaxAdapter:
    """JAX EggRoll env adapter: Isaac Lab rollouts via host callbacks."""

    def __init__(
        self,
        env_name: str,
        *,
        jax,
        jnp,
        problem_seed: int = 0,
        device: str | None = None,
        max_steps: int = 1000,
        headless: bool = True,
        launcher_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not is_isaaclab_env_tag(str(env_name)):
            raise ValueError(f"IsaacLabJaxAdapter requires an isaaclab: env tag (got {env_name!r}).")
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
        self._reset_fn = None
        self._step_fn = None
        self._ensure_host()
        self.observation_space, self.action_space = gymnax_spaces_from_host(self._host, jax=jax, jnp=jnp)
        self._obs_dim = int(self.observation_space.shape[0])
        self._reset_fn = _make_reset_fn(self._host)
        self._step_fn = _make_step_fn(self._host, self._max_steps)

    def close(self) -> None:
        if self._host is not None:
            self._host.close()
            self._host = None

    def reset(self, key: Any) -> tuple[Any, IsaacLabJaxState]:
        seed = self._jax.random.randint(key, (), 0, 2**31 - 1, dtype=self._jnp.int32)
        obs = self._reset_obs(seed)
        return obs, IsaacLabJaxState(episode_step=self._jnp.asarray(0, dtype=self._jnp.int32))

    def step(self, key: Any, state: IsaacLabJaxState, action: Any) -> core.JaxStepResult:
        del key
        obs, reward, terminated, truncated, next_step = self._step_obs(action, state.episode_step)
        return core.JaxStepResult(
            obs=obs,
            state=IsaacLabJaxState(episode_step=next_step),
            reward=reward.astype(self._jnp.float32),
            terminated=terminated.astype(self._jnp.float32),
            truncated=truncated.astype(self._jnp.float32),
            info={},
        )

    def clip_action(self, action: Any) -> Any:
        return core._clip_box_action(self.action_space, self._jnp, action)

    def _ensure_host(self) -> None:
        if self._host is not None:
            return
        self._host = make_isaaclab_env(
            self._env_name,
            headless=self._headless,
            num_envs=1,
            device=self._device,
            launcher_kwargs=self._launcher_kwargs,
            seed=self._problem_seed,
        )

    def _reset_obs(self, seed: Any) -> Any:
        if getattr(self._jnp, "__name__", "") == "numpy":
            return self._jnp.asarray(self._reset_fn(int(seed)), dtype=self._jnp.float32)
        return self._jax.pure_callback(
            self._reset_fn,
            self._jax.ShapeDtypeStruct((self._obs_dim,), self._jnp.float32),
            self._jnp.asarray(seed, dtype=self._jnp.int32),
            vmap_method="sequential",
        )

    def _step_obs(self, action: Any, episode_step: Any) -> tuple[Any, Any, Any, Any, Any]:
        action_f32 = self._jnp.asarray(action, dtype=self._jnp.float32)
        step_i32 = self._jnp.asarray(episode_step, dtype=self._jnp.int32)
        if getattr(self._jnp, "__name__", "") == "numpy":
            action_np = np.asarray(action_f32, dtype=np.float32)
            obs, reward, term, trunc, next_step = self._step_fn(action_np, int(step_i32))
            return (
                self._jnp.asarray(obs, dtype=self._jnp.float32),
                self._jnp.asarray(reward, dtype=self._jnp.float32),
                self._jnp.asarray(term, dtype=self._jnp.float32),
                self._jnp.asarray(trunc, dtype=self._jnp.float32),
                self._jnp.asarray(next_step, dtype=self._jnp.int32),
            )
        out_struct = (
            self._jax.ShapeDtypeStruct((self._obs_dim,), self._jnp.float32),
            self._jax.ShapeDtypeStruct((), self._jnp.float32),
            self._jax.ShapeDtypeStruct((), self._jnp.float32),
            self._jax.ShapeDtypeStruct((), self._jnp.float32),
            self._jax.ShapeDtypeStruct((), self._jnp.int32),
        )
        return self._jax.pure_callback(
            self._step_fn,
            out_struct,
            action_f32,
            step_i32,
            vmap_method="sequential",
        )


def _make_reset_fn(host: Any):
    def _reset(seed: int) -> np.ndarray:
        return _host_reset(host, int(seed))

    return _reset


def _make_step_fn(host: Any, max_steps: int):
    def _step(action: np.ndarray, episode_step: int) -> tuple[np.ndarray, float, bool, bool, int]:
        obs, reward, term, trunc = _host_step(host, action)
        next_step = min(int(episode_step) + 1, int(max_steps))
        term = bool(term) or next_step >= int(max_steps)
        return (
            obs,
            float(reward),
            np.float32(bool(term)),
            np.float32(bool(trunc)),
            int(next_step),
        )

    return _step
