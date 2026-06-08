from types import SimpleNamespace

import pytest
import torch

from rl.core.rollout_metrics import update_onpolicy_rollout_metrics


def test_update_onpolicy_rollout_metrics_tracks_completed_vector_episodes():
    state = SimpleNamespace()
    batch = {
        ("next", "reward"): torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ),
        ("next", "done"): torch.tensor(
            [
                [False, True],
                [True, False],
            ]
        ),
    }

    metrics = update_onpolicy_rollout_metrics(state, batch, num_envs=2)

    assert metrics["rollout_reward"] == pytest.approx(2.5)
    assert metrics["nonfinite_reward_fraction"] == pytest.approx(0.0)
    assert metrics["rollout_return"] == pytest.approx(3.0)
    assert metrics["rollout_length"] == pytest.approx(1.5)
    assert state.rollout_returns.tolist() == [0.0, 4.0]
    assert state.rollout_lengths.tolist() == [0.0, 1.0]


def test_update_onpolicy_rollout_metrics_reports_reward_before_episode_done():
    state = SimpleNamespace()
    batch = {
        ("next", "reward"): torch.tensor([[0.25, 0.75]]),
        ("next", "done"): torch.tensor([[False, False]]),
    }

    metrics = update_onpolicy_rollout_metrics(state, batch, num_envs=2)

    assert metrics == {
        "rollout_reward": pytest.approx(0.5),
        "nonfinite_reward_fraction": pytest.approx(0.0),
        "rollout_return": None,
        "rollout_length": None,
    }


def test_update_onpolicy_rollout_metrics_ignores_nonfinite_rewards_for_returns():
    state = SimpleNamespace()
    batch = {
        ("next", "reward"): torch.tensor([[float("nan"), 1.0], [2.0, float("inf")]]),
        ("next", "done"): torch.tensor([[False, False], [True, True]]),
    }

    metrics = update_onpolicy_rollout_metrics(state, batch, num_envs=2)

    assert metrics["rollout_reward"] == pytest.approx(1.5)
    assert metrics["nonfinite_reward_fraction"] == pytest.approx(0.5)
    assert metrics["rollout_return"] == pytest.approx(1.5)
    assert metrics["rollout_length"] == pytest.approx(2.0)
