from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rl.core.episode_rollout import (
    MeanReturnResult,
    Trajectory,
    _bool_any,
    _float_sum,
    _obs_for_policy,
    _resolve_max_episode_steps,
    _scale_action_to_space,
    _unpack_reset_result,
    _unpack_step_result,
    collect_denoised_trajectory,
    collect_episode_return,
    collect_trajectory_with_noise,
    evaluate_for_best,
    mean_return_over_runs,
)


class _ContinuousActionSpace:
    def __init__(self):
        self.low = np.asarray([-2.0, -1.0], dtype=np.float32)
        self.high = np.asarray([2.0, 3.0], dtype=np.float32)


class _DiscreteActionSpace:
    def __init__(self, n: int):
        self.n = int(n)


class _FakeEnv:
    def __init__(self, *, action_space, rewards: list[float], term_at: int):
        self.action_space = action_space
        self._rewards = list(rewards)
        self._term_at = int(term_at)
        self.actions_seen: list[object] = []
        self.step_count = 0

    def reset(self, seed=None):
        _ = seed
        self.step_count = 0
        return np.asarray([0.0, 0.0], dtype=np.float32), {"ok": True}

    def step(self, action):
        self.actions_seen.append(action)
        self.step_count += 1
        idx = min(self.step_count - 1, len(self._rewards) - 1)
        reward = float(self._rewards[idx])
        terminated = self.step_count >= self._term_at
        return np.asarray([0.0, 0.0], dtype=np.float32), reward, terminated, False, {}

    def close(self):
        return None


def test_episode_rollout_helper_paths():
    conf = SimpleNamespace(gym_conf=SimpleNamespace(max_steps=7), max_steps=5)
    assert _resolve_max_episode_steps(conf) == 7
    assert _resolve_max_episode_steps(SimpleNamespace(max_steps=9)) == 9

    assert np.allclose(_obs_for_policy(np.asarray([1.0, 2.0])), np.asarray([1.0, 2.0]))
    assert np.allclose(
        _obs_for_policy({"pixels": np.asarray([[1.0]], dtype=np.float32), "state": np.asarray([2.0], dtype=np.float32)}),
        np.asarray([[1.0]], dtype=np.float32),
    )
    merged = _obs_for_policy({"b": np.asarray([2.0]), "a": np.asarray([1.0])})
    assert np.allclose(merged, np.asarray([1.0, 2.0], dtype=np.float32))

    assert _bool_any(True) is True
    assert _bool_any(np.asarray([0, 0, 1])) is True
    assert _float_sum(3) == 3.0
    assert _float_sum(np.asarray([1.0, 2.0])) == 3.0

    discrete_action = _scale_action_to_space(np.asarray([2.0]), _DiscreteActionSpace(3))
    assert discrete_action == 2
    cont_action = _scale_action_to_space(np.asarray([-1.0, 1.0]), _ContinuousActionSpace())
    assert np.allclose(cont_action, np.asarray([-2.0, 3.0], dtype=np.float64))

    obs, info = _unpack_reset_result((np.asarray([1.0]), {"k": 1}))
    assert np.allclose(obs, np.asarray([1.0]))
    assert info == {"k": 1}
    obs2, info2 = _unpack_reset_result(np.asarray([2.0]))
    assert np.allclose(obs2, np.asarray([2.0]))
    assert info2 == {}

    s5 = _unpack_step_result((0, 1.0, np.asarray([True]), np.asarray([False]), {}))
    assert s5[2] is True
    s4 = _unpack_step_result((0, 1.0, np.asarray([True]), {}))
    assert s4[2] is True
    with pytest.raises(ValueError, match="Unsupported env.step"):
        _ = _unpack_step_result((1, 2, 3))


def test_collect_episode_and_denoise_paths(monkeypatch):
    fake_env = _FakeEnv(action_space=_ContinuousActionSpace(), rewards=[1.0, 2.0, 3.0], term_at=2)
    env_conf = SimpleNamespace(make=lambda: fake_env, gym_conf=SimpleNamespace(max_steps=10), noise_seed_0=10, frozen_noise=False)

    policy_calls: list[np.ndarray] = []

    def _policy(obs):
        policy_calls.append(np.asarray(obs))
        return np.asarray([-1.0, 1.0], dtype=np.float32)

    ret = collect_episode_return(env_conf, _policy, noise_seed=7)
    assert ret == 3.0
    assert len(policy_calls) == 2
    assert np.allclose(fake_env.actions_seen[0], np.asarray([-2.0, 3.0], dtype=np.float64))

    traj, used_seed = collect_trajectory_with_noise(env_conf, _policy, i_noise=5, denoise_seed=2)
    assert traj.rreturn == 3.0
    assert used_seed == 17

    monkeypatch.setattr(
        "rl.core.episode_rollout.collect_trajectory_with_noise",
        lambda _env_conf, _policy, *, i_noise=None, denoise_seed=0: (Trajectory(float(denoise_seed + 1)), int(denoise_seed)),
    )
    mr = mean_return_over_runs(env_conf, _policy, 3, i_noise=9)
    assert mr.mean == pytest.approx(2.0)
    assert mr.all_same is False
    with pytest.raises(ValueError, match="num_denoise must be > 0"):
        _ = mean_return_over_runs(env_conf, _policy, 0)

    monkeypatch.setattr(
        "rl.core.episode_rollout.mean_return_over_runs",
        lambda *_args, **_kwargs: MeanReturnResult(mean=4.0, se=0.5, all_same=False),
    )
    traj_none, seed_none = collect_denoised_trajectory(env_conf, _policy, num_denoise=None, i_noise=1)
    assert seed_none is not None
    traj_one, seed_one = collect_denoised_trajectory(env_conf, _policy, num_denoise=1, i_noise=1)
    assert seed_one is not None
    traj_many, seed_many = collect_denoised_trajectory(env_conf, _policy, num_denoise=3, i_noise=1)
    assert traj_many.rreturn == 4.0
    assert traj_many.rreturn_se == 0.5
    assert seed_many is None

    env_conf_frozen = SimpleNamespace(make=lambda: fake_env, gym_conf=SimpleNamespace(max_steps=10), noise_seed_0=0, frozen_noise=True)
    traj_frozen, _ = collect_denoised_trajectory(env_conf_frozen, _policy, num_denoise=3, i_noise=1)
    assert traj_frozen.rreturn_se is None

    best = evaluate_for_best(env_conf, _policy, 3, i_noise=999)
    assert best == 4.0
