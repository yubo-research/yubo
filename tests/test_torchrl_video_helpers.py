from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from rl.algos.torchrl_video import (
    render_best_policy_videos,
    resolve_max_episode_steps,
    rollout_episode,
    scale_action_to_space,
    select_video_episode_indices,
)


class _BoxSpaceStub:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


class _EnvStub:
    def __init__(self):
        self.action_space = _BoxSpaceStub(low=[-2.0, -2.0], high=[2.0, 2.0])
        self.actions = []
        self.step_count = 0

    def reset(self, *, seed=None):
        _ = seed
        return np.asarray([0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.step_count += 1
        terminated = self.step_count >= 1
        truncated = False
        return np.asarray([0.0, 0.0], dtype=np.float32), 1.0, terminated, truncated, {}

    def close(self):
        return


class _EnvConfStub:
    def __init__(self, *, max_steps=5, gym_conf=True):
        self.gym_conf = SimpleNamespace(max_steps=max_steps) if gym_conf else None
        self.made_envs = []

    def make(self, **_kwargs):
        env = _EnvStub()
        self.made_envs.append(env)
        return env


def test_scale_action_to_space_and_resolve_steps():
    action = np.asarray([-1.0, 1.0], dtype=np.float32)
    scaled = scale_action_to_space(action, _BoxSpaceStub(low=[-2.0, -2.0], high=[2.0, 2.0]))
    assert np.allclose(scaled, np.asarray([-2.0, 2.0], dtype=np.float32))

    no_bounds = scale_action_to_space(action, object())
    assert np.allclose(no_bounds, action)

    assert resolve_max_episode_steps(_EnvConfStub(max_steps=17)) == 17
    assert resolve_max_episode_steps(_EnvConfStub(gym_conf=False)) == 99999


def test_select_video_episode_indices_modes():
    returns = [1.0, 3.0, 2.0, 0.5]
    assert select_video_episode_indices(returns, selection="best", num_video_episodes=2, base_seed=1) == [1, 2]
    assert select_video_episode_indices(returns, selection="first", num_video_episodes=2, base_seed=1) == [0, 1]

    random_indices_1 = select_video_episode_indices(returns, selection="random", num_video_episodes=2, base_seed=42)
    random_indices_2 = select_video_episode_indices(returns, selection="random", num_video_episodes=2, base_seed=42)
    assert random_indices_1 == random_indices_2
    assert len(random_indices_1) == 2


def test_rollout_episode_runs_and_scales_action(tmp_path):
    env_conf = _EnvConfStub(max_steps=3)

    def _policy(_state):
        return np.asarray([-1.0, 1.0], dtype=np.float32)

    total_return = rollout_episode(
        env_conf,
        _policy,
        seed=0,
        render_video=False,
        video_dir=Path(tmp_path),
        video_prefix="vid",
    )
    assert total_return == 1.0
    assert len(env_conf.made_envs) == 1
    assert np.allclose(env_conf.made_envs[0].actions[0], np.asarray([-2.0, 2.0], dtype=np.float32))


def test_render_best_policy_videos_selection_and_guards(monkeypatch, tmp_path):
    config = SimpleNamespace(
        video_enable=True,
        video_episode_selection="best",
        video_seed_base=10,
        eval_seed_base=None,
        seed=7,
        video_num_episodes=3,
        video_num_video_episodes=1,
        video_dir=str(tmp_path / "videos"),
        video_prefix="ppo",
        env_tag="pend",
    )
    env_setup = SimpleNamespace(problem_seed=1, noise_seed_0=2)
    modules = SimpleNamespace()
    training_setup = SimpleNamespace(exp_dir=Path(tmp_path))
    train_state = SimpleNamespace(best_actor_state={"best": 1})

    rollout_calls = []
    returns_by_seed = {10: 1.0, 11: 5.0, 12: 2.0}

    def _fake_rollout(env_conf, policy, *, seed, render_video, video_dir, video_prefix):
        _ = env_conf, policy, video_dir, video_prefix
        rollout_calls.append((seed, render_video))
        if render_video:
            return 0.0
        return returns_by_seed[seed]

    monkeypatch.setattr("rl.algos.torchrl_video.rollout_episode", _fake_rollout)

    @contextmanager
    def _temporary_actor_state(_modules, _snapshot, *, device):
        _ = device
        yield

    eval_env_conf = _EnvConfStub(max_steps=4, gym_conf=True)
    render_best_policy_videos(
        config,
        env_setup,
        modules,
        training_setup,
        train_state,
        device=torch.device("cpu"),
        build_eval_env_conf=lambda *_args: eval_env_conf,
        eval_policy_factory=lambda *_args: object(),
        capture_actor_state=lambda *_args: {"current": 1},
        temporary_actor_state=_temporary_actor_state,
    )

    non_video_calls = [entry for entry in rollout_calls if not entry[1]]
    video_calls = [entry for entry in rollout_calls if entry[1]]
    assert [seed for seed, _ in non_video_calls] == [10, 11, 12]
    assert [seed for seed, _ in video_calls] == [11]

    bad_config = SimpleNamespace(**{**config.__dict__, "video_episode_selection": "bad"})
    try:
        render_best_policy_videos(
            bad_config,
            env_setup,
            modules,
            training_setup,
            train_state,
            device=torch.device("cpu"),
            build_eval_env_conf=lambda *_args: eval_env_conf,
            eval_policy_factory=lambda *_args: object(),
            capture_actor_state=lambda *_args: {"current": 1},
            temporary_actor_state=_temporary_actor_state,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised

    non_gym_env_conf = _EnvConfStub(max_steps=4, gym_conf=False)
    render_best_policy_videos(
        config,
        env_setup,
        modules,
        training_setup,
        train_state,
        device=torch.device("cpu"),
        build_eval_env_conf=lambda *_args: non_gym_env_conf,
        eval_policy_factory=lambda *_args: object(),
        capture_actor_state=lambda *_args: {"current": 1},
        temporary_actor_state=_temporary_actor_state,
    )

