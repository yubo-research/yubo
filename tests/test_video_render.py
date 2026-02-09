from pathlib import Path

import numpy as np

from experiments.video_render import _rollout_episode, render_policy_videos


class _DummySpace:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


class _DummyGymConf:
    def __init__(self, low, high, *, transform_state: bool):
        self.state_space = _DummySpace(low, high)
        self.transform_state = transform_state
        self.max_steps = 5


class _DummyEnvConf:
    def __init__(self, low, high, *, transform_state: bool = True):
        self.gym_conf = _DummyGymConf(low, high, transform_state=transform_state)
        self.ensure_spaces_called = 0

    def ensure_spaces(self):
        self.ensure_spaces_called += 1


def test_render_policy_videos_selects_best(monkeypatch, tmp_path):
    calls = []
    rollout_returns = {11: 1.0, 12: 3.0, 13: 2.0}

    def _fake_rollout(*args, **kwargs):
        _ = args
        calls.append(kwargs)
        if kwargs["render_video"]:
            return 0.0
        return rollout_returns[kwargs["seed"]]

    monkeypatch.setattr("experiments.video_render._rollout_episode", _fake_rollout)

    env_conf = _DummyEnvConf(low=[0.0, 0.0], high=[1.0, 1.0], transform_state=True)
    render_policy_videos(
        env_conf,
        policy=object(),
        video_dir=tmp_path,
        video_prefix="bo",
        num_episodes=3,
        num_video_episodes=1,
        episode_selection="best",
        seed_base=11,
    )

    assert env_conf.ensure_spaces_called == 1
    eval_calls = [call for call in calls if not call["render_video"]]
    video_calls = [call for call in calls if call["render_video"]]
    assert len(eval_calls) == 3
    assert len(video_calls) == 1
    assert video_calls[0]["seed"] == 12
    assert video_calls[0]["video_prefix"] == "bo_ep001"


def test_render_policy_videos_handles_non_finite_bounds(monkeypatch, tmp_path):
    calls = []

    def _fake_rollout(*args, **kwargs):
        _ = args
        calls.append(kwargs)
        return 0.0

    monkeypatch.setattr("experiments.video_render._rollout_episode", _fake_rollout)

    env_conf = _DummyEnvConf(
        low=[-np.inf, 0.0, np.nan],
        high=[np.inf, 1.0, np.nan],
        transform_state=True,
    )
    render_policy_videos(
        env_conf,
        policy=object(),
        video_dir=Path(tmp_path),
        video_prefix="bo",
        num_episodes=2,
        num_video_episodes=1,
        episode_selection="first",
        seed_base=3,
    )

    assert len(calls) == 3


class _DummyVideoEnv:
    def __init__(self):
        self.action_space = _DummySpace(low=[-1.0, -1.0], high=[1.0, 1.0])
        self._steps = 0

    def reset(self, *, seed=None):
        _ = seed
        return np.asarray([-2.0, 0.5, 2.0], dtype=np.float32), {}

    def step(self, action):
        _ = action
        self._steps += 1
        done = self._steps >= 1
        return np.asarray([-2.0, 0.5, 2.0], dtype=np.float32), 1.0, done, False, {}

    def close(self):
        return


class _DummyVideoEnvConf:
    def __init__(self):
        self.env_name = "HalfCheetah-v5"
        self.gym_conf = type("GC", (), {"max_steps": 2})()

    def make(self, **kwargs):
        _ = kwargs
        return _DummyVideoEnv()


def test_rollout_uses_raw_state_when_transform_disabled():
    captured = {}

    def _policy(state):
        captured["state"] = np.asarray(state, dtype=np.float32)
        return np.asarray([0.0, 0.0], dtype=np.float32)

    ret = _rollout_episode(
        _DummyVideoEnvConf(),
        _policy,
        seed=0,
        transform_state=False,
        lb=None,
        width=None,
        render_video=False,
        video_dir=None,
        video_prefix="x",
    )
    assert np.allclose(captured["state"], np.asarray([-2.0, 0.5, 2.0], dtype=np.float32))
    assert ret == 1.0
