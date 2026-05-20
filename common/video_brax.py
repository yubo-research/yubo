from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from common.video_batch import (
    _is_headless_video_error,
    _temporary_mujoco_gl,
    _video_gl_candidates,
    select_video_episode_indices,
)
from common.video_rollout import _frame_to_uint8, _open_frame_video_writer
from common.video_spaces import resolve_max_episode_steps
from optimizer.eggroll_runtime_core import EggRollActionSelector, IdentityNoiser, require_eggroll_jax_stack


def is_brax_env_conf(env_conf: Any) -> bool:
    return str(getattr(env_conf, "env_name", "")).startswith("brax:")


def render_brax_policy_videos(
    env_conf: Any,
    policy: Any,
    *,
    video_dir: Path | str,
    video_prefix: str,
    num_episodes: int,
    num_video_episodes: int,
    episode_selection: str,
    seed_base: int,
) -> bool:
    if not is_brax_env_conf(env_conf):
        return False
    if not hasattr(policy, "model_cls") or not hasattr(policy, "params"):
        raise RuntimeError("Brax video capture currently requires an EggRoll JAX policy.")
    renderer = _BraxPolicyRenderer(env_conf, policy)
    selected = _selected_episode_indices(renderer, int(num_episodes), int(num_video_episodes), str(episode_selection), int(seed_base))
    video_path = Path(video_dir)
    video_path.mkdir(parents=True, exist_ok=True)
    print(f"[video] brax dir={video_path} episodes={num_episodes} videos={len(selected)} select={episode_selection}", flush=True)
    _render_selected(renderer, selected, video_path, str(video_prefix), int(seed_base))
    return True


def _selected_episode_indices(renderer, num_episodes: int, num_video_episodes: int, selection: str, seed_base: int) -> list[int]:
    if selection == "best" and num_episodes <= num_video_episodes:
        return list(range(max(num_episodes, 0)))
    if selection == "best":
        returns = [renderer.rollout(seed=seed_base + idx, video_file=None) for idx in range(num_episodes)]
        return select_video_episode_indices(
            returns,
            selection=selection,
            num_video_episodes=num_video_episodes,
            base_seed=seed_base,
        )
    if selection == "random":
        return select_video_episode_indices(
            [0.0] * int(num_episodes),
            selection=selection,
            num_video_episodes=num_video_episodes,
            base_seed=seed_base,
        )
    return list(range(min(num_video_episodes, num_episodes)))


def _render_selected(renderer, selected: list[int], video_dir: Path, video_prefix: str, seed_base: int) -> None:
    preferred = None
    for episode_idx in selected:
        rendered, preferred = _render_one_with_backend(
            renderer,
            seed=seed_base + int(episode_idx),
            video_file=video_dir / f"{video_prefix}_ep{episode_idx:03d}-episode-0.mp4",
            preferred_gl=preferred,
        )
        if not rendered:
            return


def _render_one_with_backend(renderer, *, seed: int, video_file: Path, preferred_gl: str | None):
    backends = [preferred_gl] if preferred_gl is not None else _video_gl_candidates()
    last_headless_exc = None
    for backend in backends:
        try:
            with _temporary_mujoco_gl(backend):
                renderer.rollout(seed=seed, video_file=video_file)
            if preferred_gl is None and backend is not None:
                print(f"[video] using MUJOCO_GL={backend}", flush=True)
            return True, backend
        except Exception as exc:
            if not _is_headless_video_error(exc):
                raise
            last_headless_exc = exc
    print(f"[video] skipping Brax video capture: headless/OpenGL context unavailable ({last_headless_exc})", flush=True)
    return False, preferred_gl


class _BraxPolicyRenderer:
    def __init__(self, env_conf: Any, policy: Any) -> None:
        jax, jnp, simple_es_tree_key = require_eggroll_jax_stack()
        from brax import envs

        self.jax = jax
        self.jnp = jnp
        self.policy = policy
        self.env = envs.get_environment(str(env_conf.env_name).split(":", 1)[1])
        self.noiser = IdentityNoiser()
        self.selector = EggRollActionSelector(jax, jnp, deterministic_policy=bool(getattr(policy, "_eggroll_deterministic_policy", False)))
        key = jax.random.key(int(getattr(policy, "problem_seed", 0) or 0) & 0xFFFFFFFF)
        self.es_tree_key = simple_es_tree_key(policy.params, jax.random.fold_in(key, 1), policy.scan_map)
        self.steps = _resolve_steps(env_conf, policy, self.env)
        self.step_once = jax.jit(self._make_step_once())

    def rollout(self, *, seed: int, video_file: Path | None) -> float:
        state, key = self._reset(seed)
        trajectory = [state.pipeline_state] if video_file is not None else None
        total_return = 0.0
        for _ in range(self.steps):
            state, key = self.step_once(state, key)
            total_return += float(np.asarray(state.reward))
            if trajectory is not None:
                trajectory.append(state.pipeline_state)
            if bool(np.asarray(state.done)):
                break
        if video_file is not None:
            self._write_video(video_file, trajectory or [])
        return float(total_return)

    def _reset(self, seed: int):
        key = self.jax.random.key(int(seed) & 0xFFFFFFFF)
        reset_key, rollout_key = self.jax.random.split(key)
        return self.env.reset(reset_key), rollout_key

    def _make_step_once(self):
        def step_once(state, key):
            key, action_key = self.jax.random.split(key)
            policy_dist = self.policy.model_cls.forward(
                self.noiser,
                None,
                None,
                self.policy.frozen_params,
                self.policy.params,
                self.es_tree_key,
                None,
                state.obs,
            )
            action = self.jnp.clip(self.selector.select_action(policy_dist, action_key), -1.0, 1.0)
            return self.env.step(state, action), key

        return step_once

    def _write_video(self, video_file: Path, trajectory: list[Any]) -> None:
        from brax.io import image

        frames = image.render_array(self.env.sys, trajectory, height=240, width=320)
        writer = _open_frame_video_writer(video_file, fps=30)
        try:
            for frame in frames:
                writer.append_data(_frame_to_uint8(frame))
        finally:
            writer.close()


def _resolve_steps(env_conf: Any, policy: Any, env: Any) -> int:
    policy_steps = getattr(policy, "_eggroll_steps_per_episode", None)
    if policy_steps is not None:
        return max(int(policy_steps), 1)
    env_steps = getattr(env, "episode_length", None)
    if env_steps is not None:
        return max(int(env_steps), 1)
    return min(resolve_max_episode_steps(env_conf), 1000)
