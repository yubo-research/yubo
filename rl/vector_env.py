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


def _make_dm_control_env(*, env_name: str, env_kwargs: dict, obs_mode: str, buf=None, seed=0):
    import gymnasium as gym
    import numpy as np
    import pufferlib

    from problems.dm_control_env import make as make_dm_control_env

    class _DMControlPufferEnv(pufferlib.PufferEnv):
        def __init__(self, *, buf=None, seed=0):
            kwargs = dict(env_kwargs)
            self._env = make_dm_control_env(
                env_name,
                render_mode="rgb_array",
                obs_mode=str(obs_mode),
                **kwargs,
            )
            obs_space = self._env.observation_space
            act_space = self._env.action_space
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
            obs, info = self._env.reset(seed=seed)
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
            return self.observations, info

        def step(self, actions):
            if not self._initialized:
                raise pufferlib.APIUsageError("step() called before reset()")
            if self._done:
                self.reset()
                return self.observations, float(self.rewards[0]), False, False, {}
            action = np.asarray(actions, dtype=self.single_action_space.dtype)
            action = np.ravel(action).reshape(self.single_action_space.shape)
            action = np.clip(action, self.single_action_space.low, self.single_action_space.high)
            obs, reward, done, truncated, info = self._env.step(action)
            obs_arr = np.asarray(obs, dtype=self.single_observation_space.dtype)
            reward_f = float(reward)
            done_b = bool(done)
            trunc_b = bool(truncated)
            self.observations[0] = obs_arr
            self.rewards[0] = reward_f
            self.terminals[0] = done_b
            self.truncations[0] = trunc_b
            self.masks[0] = True
            self._episode_return += reward_f
            self._episode_length += 1
            if done_b or trunc_b:
                info = dict(info)
                info["episode_return"] = float(self._episode_return)
                info["episode_length"] = int(self._episode_length)
                self._episode_return = 0.0
                self._episode_length = 0
            self._done = bool(done_b or trunc_b)
            return self.observations, reward_f, done_b, trunc_b, info

        def close(self):
            return self._env.close()

    return _DMControlPufferEnv(buf=buf, seed=seed)


def _make_gymnasium_env(
    *,
    env_name: str,
    env_kwargs: dict,
    render_mode="rgb_array",
    buf=None,
    seed=0,
    preprocess_spec=None,
):
    import gymnasium as gym
    import pufferlib
    import pufferlib.emulation

    from common.env_preprocessing import apply_gym_preprocessing

    kwargs = dict(env_kwargs)
    try:
        env = gym.make(env_name, render_mode=render_mode, **kwargs)
    except TypeError:
        env = gym.make(env_name, **kwargs)
    if isinstance(env.action_space, gym.spaces.Box):
        env = pufferlib.ClipAction(env)
    if preprocess_spec is not None and preprocess_spec.enabled:
        env = apply_gym_preprocessing(env, preprocess_spec=preprocess_spec)
    env = pufferlib.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf, seed=seed)


def _build_gymnasium_env_creator(*, env_name: str, env_kwargs: dict, preprocess_spec=None, pufferlib):
    _ = pufferlib
    return functools.partial(
        _make_gymnasium_env,
        env_name=env_name,
        env_kwargs=env_kwargs,
        preprocess_spec=preprocess_spec,
    )


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
    env_name, env_kwargs = resolve_gym_env_name_fn(env_tag)
    if str(env_name).startswith("dm_control/"):
        return (
            functools.partial(
                _make_dm_control_env,
                env_name=str(env_name),
                env_kwargs=dict(env_kwargs),
                obs_mode=str(getattr(config, "obs_mode", "vector")),
            ),
            {},
        )
    env_creator = _build_gymnasium_env_creator(
        env_name=str(env_name),
        env_kwargs=dict(env_kwargs),
        preprocess_spec=None,
        pufferlib=pufferlib,
    )
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
