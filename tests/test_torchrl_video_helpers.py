from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from common.video import (
    RLVideoContext,
    policy_for_bo_rollout,
    render_policy_videos,
    render_policy_videos_bo,
    render_policy_videos_rl,
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


class _NonGymRenderEnv:
    def __init__(self):
        self.action_space = _BoxSpaceStub(low=[-1.0, -1.0], high=[1.0, 1.0])
        self.observation_space = _BoxSpaceStub(low=[-1.0, -1.0], high=[1.0, 1.0])
        self._step_count = 0

    def reset(self, *, seed=None):
        _ = seed
        self._step_count = 0
        return np.asarray([0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        _ = action
        self._step_count += 1
        terminated = self._step_count >= 1
        return np.asarray([0.0, 0.0], dtype=np.float32), 1.0, terminated, False, {}

    def render(self):
        pixel = np.uint8(min(255, 40 * (self._step_count + 1)))
        frame = np.full((8, 8, 3), pixel, dtype=np.uint8)
        return frame

    def close(self):
        return


class _NonGymEnvConfStub:
    def __init__(self, *, max_steps=2):
        self.gym_conf = None
        self.max_steps = int(max_steps)

    def make(self, **_kwargs):
        return _NonGymRenderEnv()


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


def test_render_policy_videos_rl_selection_and_guards(monkeypatch, tmp_path):
    config = SimpleNamespace(
        video_enable=True,
        video_episode_selection="best",
        video_seed_base=10,
        eval_seed_base=None,
        seed=7,
        video_num_episodes=3,
        video_num_video_episodes=1,
        video_prefix="ppo",
        env_tag="pend",
    )
    env_setup = SimpleNamespace(problem_seed=1, noise_seed_0=2)
    modules = SimpleNamespace()
    training_setup = SimpleNamespace(exp_dir=Path(tmp_path))
    train_state = SimpleNamespace(best_actor_state={"best": 1})

    rollout_calls = []
    rollout_video_dirs = []
    returns_by_seed = {10: 1.0, 11: 5.0, 12: 2.0}

    def _fake_rollout(env_conf, policy, *, seed, render_video, video_dir, video_prefix):
        _ = env_conf, policy, video_dir, video_prefix
        rollout_calls.append((seed, render_video))
        rollout_video_dirs.append(video_dir)
        if render_video:
            return 0.0
        return returns_by_seed[seed]

    monkeypatch.setattr("common.video.rollout_episode", _fake_rollout)

    @contextmanager
    def _with_actor_state(_modules, _snapshot, *, device):
        _ = device
        yield

    eval_env_conf = _EnvConfStub(max_steps=4, gym_conf=True)
    ctx = RLVideoContext(
        build_eval_env_conf=lambda *_args: eval_env_conf,
        make_eval_policy=lambda *_args: object(),
        capture_actor_state=lambda *_args: {"current": 1},
        with_actor_state=_with_actor_state,
    )
    render_policy_videos_rl(
        config,
        env_setup,
        modules,
        training_setup,
        train_state,
        ctx,
        device=torch.device("cpu"),
    )

    non_video_calls = [entry for entry in rollout_calls if not entry[1]]
    video_calls = [entry for entry in rollout_calls if entry[1]]
    assert [seed for seed, _ in non_video_calls] == [10, 11, 12]
    assert [seed for seed, _ in video_calls] == [11]
    assert rollout_video_dirs[0] is None
    assert str(rollout_video_dirs[-1]).endswith("videos")

    bad_config = SimpleNamespace(**{**config.__dict__, "video_episode_selection": "bad"})
    try:
        render_policy_videos_rl(
            bad_config,
            env_setup,
            modules,
            training_setup,
            train_state,
            ctx,
            device=torch.device("cpu"),
        )
        raised = False
    except ValueError:
        raised = True
    assert raised

    non_gym_ctx = RLVideoContext(
        build_eval_env_conf=lambda *_args: _EnvConfStub(max_steps=4, gym_conf=False),
        make_eval_policy=lambda *_args: object(),
        capture_actor_state=lambda *_args: {"current": 1},
        with_actor_state=_with_actor_state,
    )
    render_policy_videos_rl(
        config,
        env_setup,
        modules,
        training_setup,
        train_state,
        non_gym_ctx,
        device=torch.device("cpu"),
    )


def test_render_policy_videos_non_gym_not_skipped(monkeypatch, tmp_path):
    env_conf = _NonGymEnvConfStub(max_steps=2)

    calls = []

    def _fake_rollout(_env_conf, _policy, *, seed, render_video, video_dir, video_prefix):
        calls.append((seed, render_video, video_dir, video_prefix))
        return 1.0

    monkeypatch.setattr("common.video.rollout_episode", _fake_rollout)

    render_policy_videos(
        env_conf,
        lambda _s: np.asarray([0.0, 0.0], dtype=np.float32),
        video_dir=Path(tmp_path) / "videos",
        video_prefix="non_gym",
        num_episodes=2,
        num_video_episodes=1,
        episode_selection="best",
        seed_base=17,
    )

    assert len(calls) == 3
    non_video_calls = [entry for entry in calls if entry[1] is False]
    video_calls = [entry for entry in calls if entry[1] is True]
    assert len(non_video_calls) == 2
    assert len(video_calls) == 1


def test_policy_for_bo_rollout_no_transform():
    class _EnvConfNoTransform:
        def ensure_spaces(self):
            self.gym_conf = None

    env_conf = _EnvConfNoTransform()
    env_conf.ensure_spaces()

    def policy(s):
        return np.array([0.5, -0.5], dtype=np.float32)

    wrapped = policy_for_bo_rollout(env_conf, policy)
    out = wrapped(np.array([1.0, 2.0]))
    assert np.allclose(out, [0.5, -0.5])


def test_policy_for_bo_rollout_with_transform():
    class _SpaceStub:
        low = np.array([0.0, 0.0])
        high = np.array([2.0, 4.0])

    class _EnvConfTransform:
        def ensure_spaces(self):
            self.gym_conf = SimpleNamespace(transform_state=True, state_space=_SpaceStub())

    env_conf = _EnvConfTransform()
    env_conf.ensure_spaces()

    def policy(s):
        return np.array([s[0] + s[1]], dtype=np.float32)  # pass-through sum

    wrapped = policy_for_bo_rollout(env_conf, policy)
    # state [1, 2] -> norm (1-0)/2, (2-0)/4 = [0.5, 0.5] -> policy returns [1.0]
    out = wrapped(np.array([1.0, 2.0]))
    assert np.allclose(out, [1.0])


def test_policy_for_bo_rollout_with_partial_infinite_bounds():
    class _SpaceStub:
        low = np.array([0.0, -np.inf], dtype=np.float32)
        high = np.array([2.0, np.inf], dtype=np.float32)

    class _EnvConfTransform:
        def ensure_spaces(self):
            self.gym_conf = SimpleNamespace(transform_state=True, state_space=_SpaceStub())

    env_conf = _EnvConfTransform()
    env_conf.ensure_spaces()

    captured = {}

    def policy(s):
        captured["state"] = np.asarray(s, dtype=np.float32)
        return np.array([0.0], dtype=np.float32)

    wrapped = policy_for_bo_rollout(env_conf, policy)
    _ = wrapped(np.array([1.0, 3.0], dtype=np.float32))
    assert np.allclose(captured["state"], np.array([0.5, 3.0], dtype=np.float32))


def test_policy_for_bo_rollout_with_all_infinite_bounds_sanitizes_nan_input():
    class _SpaceStub:
        low = np.array([-np.inf, -np.inf], dtype=np.float32)
        high = np.array([np.inf, np.inf], dtype=np.float32)

    class _EnvConfTransform:
        def ensure_spaces(self):
            self.gym_conf = SimpleNamespace(transform_state=True, state_space=_SpaceStub())

    env_conf = _EnvConfTransform()
    env_conf.ensure_spaces()

    captured = {}

    def policy(s):
        captured["state"] = np.asarray(s, dtype=np.float32)
        return np.array([0.0], dtype=np.float32)

    wrapped = policy_for_bo_rollout(env_conf, policy)
    _ = wrapped(np.array([np.nan, np.inf], dtype=np.float32))
    assert np.isfinite(captured["state"]).all()
    assert np.allclose(captured["state"], np.array([0.0, 1e6], dtype=np.float32))


def test_render_policy_videos_bo_smoke(monkeypatch, tmp_path):
    env_conf = _EnvConfStub(max_steps=2, gym_conf=True)
    env_conf.ensure_spaces = lambda: None

    def policy(s):
        return np.array([-1.0, 1.0], dtype=np.float32)

    monkeypatch.setattr("common.video.render_policy_videos", lambda *a, **kw: None)
    render_policy_videos_bo(
        env_conf,
        policy,
        video_dir=tmp_path,
        video_prefix="bo",
        num_episodes=1,
        num_video_episodes=0,
        episode_selection="first",
        seed_base=0,
    )


def test_render_policy_videos_direct(tmp_path):
    """Exercise render_policy_videos with a stub env (no video recording)."""
    env_conf = _EnvConfStub(max_steps=2, gym_conf=True)

    def policy(s):
        return np.array([-1.0, 1.0], dtype=np.float32)

    video_dir = Path(tmp_path) / "videos"
    render_policy_videos(
        env_conf,
        policy,
        video_dir=video_dir,
        video_prefix="test",
        num_episodes=2,
        num_video_episodes=0,
        episode_selection="first",
        seed_base=0,
    )
    assert video_dir.exists()


def test_rollout_episode_records_non_gym_video(monkeypatch, tmp_path):
    env_conf = _NonGymEnvConfStub(max_steps=2)

    frames = []
    video_path_holder = {}

    def _fake_writer(video_path, *, fps=30):
        _ = fps
        path = Path(video_path)
        video_path_holder["path"] = path

        class _Writer:
            def append_data(self, frame):
                frames.append(np.asarray(frame))

            def close(self):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

        return _Writer()

    monkeypatch.setattr("common.video._open_frame_video_writer", _fake_writer)

    ret = rollout_episode(
        env_conf,
        lambda _s: np.asarray([0.0, 0.0], dtype=np.float32),
        seed=3,
        render_video=True,
        video_dir=Path(tmp_path) / "videos",
        video_prefix="dm_non_gym",
    )

    assert ret == 1.0
    assert video_path_holder["path"].name == "dm_non_gym-episode-0.mp4"
    assert video_path_holder["path"].exists()
    assert len(frames) >= 2
    assert frames[0].dtype == np.uint8


def test_render_policy_videos_skips_headless_video_error(monkeypatch, tmp_path, capsys):
    env_conf = _EnvConfStub(max_steps=2, gym_conf=True)

    def policy(_s):
        return np.array([0.0, 0.0], dtype=np.float32)

    def _fake_rollout(_env_conf, _policy, *, seed, render_video, video_dir, video_prefix):
        _ = seed, video_dir, video_prefix
        if render_video:
            raise RuntimeError("X11: The DISPLAY environment variable is missing")
        return 1.0

    monkeypatch.setattr("common.video.rollout_episode", _fake_rollout)
    render_policy_videos(
        env_conf,
        policy,
        video_dir=Path(tmp_path) / "videos",
        video_prefix="test",
        num_episodes=2,
        num_video_episodes=1,
        episode_selection="best",
        seed_base=0,
    )
    out = capsys.readouterr().out
    assert "skipping video capture" in out


def test_render_policy_videos_retries_with_headless_gl_backend(monkeypatch, tmp_path, capsys):
    env_conf = _EnvConfStub(max_steps=2, gym_conf=True)

    def policy(_s):
        return np.array([0.0, 0.0], dtype=np.float32)

    calls = {"render": 0}

    def _fake_rollout(_env_conf, _policy, *, seed, render_video, video_dir, video_prefix):
        _ = seed, video_dir, video_prefix
        if not render_video:
            return 1.0
        calls["render"] += 1
        if os.environ.get("MUJOCO_GL") != "egl":
            raise RuntimeError("X11: The DISPLAY environment variable is missing")
        return 0.0

    monkeypatch.setattr("common.video._video_gl_candidates", lambda: [None, "egl"])
    monkeypatch.setattr("common.video.rollout_episode", _fake_rollout)

    render_policy_videos(
        env_conf,
        policy,
        video_dir=Path(tmp_path) / "videos",
        video_prefix="test",
        num_episodes=2,
        num_video_episodes=1,
        episode_selection="best",
        seed_base=0,
    )
    out = capsys.readouterr().out
    assert calls["render"] == 2
    assert "using MUJOCO_GL=egl" in out
    assert "skipping video capture" not in out
