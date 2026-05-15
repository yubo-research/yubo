from __future__ import annotations

import functools
from typing import Any


def _vector_backend_from_name(vector_mod, name: str):
    key = str(name).strip().lower()
    if key == "serial":
        return vector_mod.Serial
    if key == "multiprocessing":
        return vector_mod.Multiprocessing
    raise ValueError("vector_backend must be one of: serial, multiprocessing")


def _build_vector_kwargs(config: Any, backend_cls, vector_mod) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if backend_cls is vector_mod.Multiprocessing:
        if getattr(config, "vector_num_workers", None) is not None:
            kwargs["num_workers"] = int(config.vector_num_workers)
        if getattr(config, "vector_batch_size", None) is not None:
            kwargs["batch_size"] = int(config.vector_batch_size)
        kwargs["overwork"] = bool(getattr(config, "vector_overwork", False))
    return kwargs


def _make_dm_control_env(
    *,
    domain: str,
    task: str,
    from_pixels: bool,
    pixels_only: bool,
    buf=None,
    seed=0,
):
    import gymnasium as gym
    import numpy as np
    import pufferlib
    from dm_control import suite

    from problems.dm_control_spaces import flatten_obs, spec_to_space

    class _DMControlPufferEnv(pufferlib.PufferEnv):
        def __init__(self, *, buf=None, seed=0):
            self._domain = domain
            self._task = task
            self._from_pixels = from_pixels
            self._pixels_only = pixels_only
            self._env = suite.load(domain, task, task_kwargs={"random": seed})
            obs_spec = self._env.observation_spec()
            act_spec = self._env.action_spec()
            obs_space = spec_to_space(obs_spec)
            act_space = spec_to_space(act_spec)

            if from_pixels:
                # Placeholder for pixel support: in a real implementation we would
                # wrap self._env with a pixel wrapper similar to the one in problems.
                # For now, we assume vector observations as per the native suite.
                pass

            self.single_observation_space = gym.spaces.Box(
                low=np.asarray(obs_space.low),
                high=np.asarray(obs_space.high),
                shape=tuple((int(v) for v in obs_space.shape)),
                dtype=np.dtype(obs_space.dtype),
            )
            self.single_action_space = gym.spaces.Box(
                low=np.asarray(act_space.low),
                high=np.asarray(act_space.high),
                shape=tuple((int(v) for v in act_space.shape)),
                dtype=np.dtype(act_space.dtype),
            )
            self.num_agents = 1
            super().__init__(buf=buf)
            self._done = True
            self._initialized = False
            self._episode_return = 0.0
            self._episode_length = 0
            if seed is not None:
                self.reset(seed=int(seed))

        def reset(self, seed=None):
            if seed is not None:
                self._env = suite.load(self._domain, self._task, task_kwargs={"random": int(seed)})
            time_step = self._env.reset()
            obs = flatten_obs(time_step.observation)
            obs_arr = np.asarray(obs, dtype=self.single_observation_space.dtype)
            self.observations[0] = obs_arr
            self.rewards[0] = 0.0
            self.terminals[0] = False
            self.truncations[0] = False
            self.masks[0] = True
            self._done = False
            self._initialized = True
            self._episode_return = 0.0
            self._episode_length = 0
            return self.observations, {}

        def step(self, actions):
            if not self._initialized:
                raise pufferlib.APIUsageError("step() called before reset()")
            if self._done:
                self.reset()
                return self.observations, float(self.rewards[0]), False, False, {}

            action = np.asarray(actions, dtype=self.single_action_space.dtype)
            action = np.ravel(action).reshape(self.single_action_space.shape)
            action = np.clip(action, self.single_action_space.low, self.single_action_space.high)

            time_step = self._env.step(action)
            obs = flatten_obs(time_step.observation)
            obs_arr = np.asarray(obs, dtype=self.single_observation_space.dtype)

            reward_f = float(time_step.reward) if time_step.reward is not None else 0.0
            done_b = bool(time_step.last())
            trunc_b = False  # dm_env doesn't distinguish truncation by default

            self.observations[0] = obs_arr
            self.rewards[0] = reward_f
            self.terminals[0] = done_b
            self.truncations[0] = trunc_b
            self.masks[0] = True

            self._episode_return += reward_f
            self._episode_length += 1

            info = {}
            if done_b or trunc_b:
                info["episode_return"] = float(self._episode_return)
                info["episode_length"] = int(self._episode_length)
                self._episode_return = 0.0
                self._episode_length = 0
            self._done = bool(done_b or trunc_b)

            return self.observations, reward_f, done_b, trunc_b, info

        def close(self):
            return self._env.close()

    return _DMControlPufferEnv(buf=buf, seed=seed)


def _make_gymnasium_env(*, env_conf: Any, render_mode="rgb_array", buf=None, seed=0):
    import pufferlib
    import pufferlib.emulation

    env = _make_runtime_env(env_conf, render_mode=render_mode)

    # We still need PufferLib-specific EpisodeStats for some backends
    env = pufferlib.EpisodeStats(env)

    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf, seed=seed)


def _make_runtime_env(env_conf: Any, *, render_mode: str | None):
    if hasattr(env_conf, "make") and callable(env_conf.make):
        return env_conf.make(render_mode=render_mode)
    if hasattr(env_conf, "make_gym_env") and callable(env_conf.make_gym_env):
        return env_conf.make_gym_env(render_mode=render_mode)
    raise TypeError(f"env_conf must expose make() or make_gym_env(), got {type(env_conf).__name__}.")


def _build_gymnasium_env_creator(*, env_conf: Any, pufferlib):
    _ = pufferlib
    return functools.partial(_make_gymnasium_env, env_conf=env_conf)


def _parse_dm_control_env_tag(env_name: str) -> tuple[str, str]:
    if not env_name.startswith("dm_control/"):
        raise ValueError(f"Expected dm_control env name, got: {env_name}")
    name = env_name.split("/", 1)[1]
    if name.endswith(("-v0", "-v1")):
        name = name.rsplit("-", 1)[0]
    if "-" not in name:
        raise ValueError(f"Expected dm_control/<domain>-<task>-v0, got: {env_name}")
    return tuple(name.split("-", 1))


def _resolve_backend(config: Any, puffer_vector):
    backend_cls = _vector_backend_from_name(puffer_vector, config.vector_backend)
    backend_kwargs = _build_vector_kwargs(config, backend_cls, puffer_vector)
    return (backend_cls, backend_kwargs)


def _resolve_env_creator(
    config: Any,
    *,
    pufferlib,
    puffer_atari,
    is_atari_env_tag_fn,
    to_puffer_game_name_fn,
    resolve_gym_env_name_fn,
):
    env_tag = str(config.env_tag)
    if is_atari_env_tag_fn(env_tag):
        game_name = to_puffer_game_name_fn(env_tag)
        return (
            puffer_atari.env_creator(game_name),
            {"framestack": int(config.framestack)},
        )

    if env_tag.startswith("dm_control/"):
        domain, task = _parse_dm_control_env_tag(env_tag)
        return (
            functools.partial(
                _make_dm_control_env,
                domain=domain,
                task=task,
                from_pixels=bool(getattr(config, "from_pixels", False)),
                pixels_only=bool(getattr(config, "pixels_only", True)),
            ),
            {},
        )

    env_creator = _build_gymnasium_env_creator(env_conf=config.env_conf, pufferlib=pufferlib)
    return (env_creator, {})


def _resolve_vector_seed(config: Any) -> int:
    if getattr(config, "problem_seed", None) is not None:
        return int(config.problem_seed)
    return int(config.seed)


def make_vector_env(
    config: Any,
    *,
    import_pufferlib_modules_fn,
    is_atari_env_tag_fn,
    to_puffer_game_name_fn,
    resolve_gym_env_name_fn,
):
    pufferlib, puffer_vector, puffer_atari = import_pufferlib_modules_fn()
    backend_cls, backend_kwargs = _resolve_backend(config, puffer_vector)
    env_creator, env_kwargs = _resolve_env_creator(
        config,
        pufferlib=pufferlib,
        puffer_atari=puffer_atari,
        is_atari_env_tag_fn=is_atari_env_tag_fn,
        to_puffer_game_name_fn=to_puffer_game_name_fn,
        resolve_gym_env_name_fn=resolve_gym_env_name_fn,
    )
    return puffer_vector.make(
        env_creator,
        env_kwargs=env_kwargs,
        backend=backend_cls,
        num_envs=int(config.num_envs),
        seed=_resolve_vector_seed(config),
        **backend_kwargs,
    )
