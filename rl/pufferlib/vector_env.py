"""Shared Puffer vector-environment factory for RL backends."""

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


def _make_gymnasium_env(*, env_name: str, env_kwargs: dict, render_mode="rgb_array", buf=None, seed=0):
    import gymnasium as gym
    import pufferlib
    import pufferlib.emulation

    kwargs = dict(env_kwargs)
    try:
        env = gym.make(env_name, render_mode=render_mode, **kwargs)
    except TypeError:
        env = gym.make(env_name, **kwargs)

    if isinstance(env.action_space, gym.spaces.Box):
        env = pufferlib.ClipAction(env)
    env = pufferlib.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf, seed=seed)


def _build_gymnasium_env_creator(env_tag: str, pufferlib, *, resolve_gym_env_name_fn):
    env_name, env_kwargs = resolve_gym_env_name_fn(env_tag)
    _ = pufferlib
    return functools.partial(_make_gymnasium_env, env_name=env_name, env_kwargs=env_kwargs)


def _resolve_backend(config: Any, puffer_vector):
    backend_cls = _vector_backend_from_name(puffer_vector, config.vector_backend)
    backend_kwargs = _build_vector_kwargs(config, backend_cls, puffer_vector)
    return backend_cls, backend_kwargs


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
        return puffer_atari.env_creator(game_name), {"framestack": int(config.framestack)}

    env_creator = _build_gymnasium_env_creator(
        env_tag,
        pufferlib,
        resolve_gym_env_name_fn=resolve_gym_env_name_fn,
    )
    return env_creator, {}


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
        seed=int(config.seed),
        **backend_kwargs,
    )
