from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torchrl_video_helper_stubs import _BoxSpaceStub, _EnvConfStub
from torchrl_video_helper_stubs_non_gym import _NonGymEnvConfStub
from torchrl_video_type_support import (
    env_conf_no_transform_instance,
    env_conf_transform_with_space,
    patch_rollout_video_writer,
)

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

    monkeypatch.setattr("common.video_batch.rollout_episode", _fake_rollout)

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

    monkeypatch.setattr("common.video_batch.rollout_episode", _fake_rollout)

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
    env_conf = env_conf_no_transform_instance()
    env_conf.ensure_spaces()

    def policy(s):
        return np.array([0.5, -0.5], dtype=np.float32)

    wrapped = policy_for_bo_rollout(env_conf, policy)
    out = wrapped(np.array([1.0, 2.0]))
    assert np.allclose(out, [0.5, -0.5])


def test_policy_for_bo_rollout_with_transform():
    env_conf = env_conf_transform_with_space(np.array([0.0, 0.0]), np.array([2.0, 4.0]))
    env_conf.ensure_spaces()

    def policy(s):
        return np.array([s[0] + s[1]], dtype=np.float32)

    wrapped = policy_for_bo_rollout(env_conf, policy)
    out = wrapped(np.array([1.0, 2.0]))
    assert np.allclose(out, [1.0])


def test_policy_for_bo_rollout_with_partial_infinite_bounds():
    env_conf = env_conf_transform_with_space(
        np.array([0.0, -np.inf], dtype=np.float32),
        np.array([2.0, np.inf], dtype=np.float32),
    )
    env_conf.ensure_spaces()

    captured = {}

    def policy(s):
        captured["state"] = np.asarray(s, dtype=np.float32)
        return np.array([0.0], dtype=np.float32)

    wrapped = policy_for_bo_rollout(env_conf, policy)
    _ = wrapped(np.array([1.0, 3.0], dtype=np.float32))
    assert np.allclose(captured["state"], np.array([0.5, 3.0], dtype=np.float32))


def test_policy_for_bo_rollout_with_all_infinite_bounds_sanitizes_nan_input():
    env_conf = env_conf_transform_with_space(
        np.array([-np.inf, -np.inf], dtype=np.float32),
        np.array([np.inf, np.inf], dtype=np.float32),
    )
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

    monkeypatch.setattr("common.video_batch.render_policy_videos", lambda *a, **kw: None)
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
    patch_rollout_video_writer(monkeypatch, frames, video_path_holder)

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

    monkeypatch.setattr("common.video_batch.rollout_episode", _fake_rollout)
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

    monkeypatch.setattr("common.video_batch._video_gl_candidates", lambda: [None, "egl"])
    monkeypatch.setattr("common.video_batch.rollout_episode", _fake_rollout)

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
