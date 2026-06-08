from __future__ import annotations

from types import SimpleNamespace

import torch

from rl.core.ppo_reward_norm import (
    _discounted_reward_returns,
    normalize_rewards_for_training,
)


def test_discounted_reward_returns_resets_on_done() -> None:
    config = SimpleNamespace(loss=SimpleNamespace(gamma=0.5))
    state = SimpleNamespace(reward_return=None)
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    dones = torch.tensor([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])

    returns = _discounted_reward_returns(config, state, rewards, dones, device=torch.device("cpu"))

    torch.testing.assert_close(
        returns,
        torch.tensor([[1.0, 2.0], [3.5, 4.0], [6.75, 8.0]]),
    )
    torch.testing.assert_close(state.reward_return, torch.tensor([0.0, 8.0]))


def test_normalize_rewards_for_training_updates_state_and_batch() -> None:
    config = SimpleNamespace(
        env=SimpleNamespace(normalize_reward=True),
        loss=SimpleNamespace(gamma=0.9),
    )
    state = SimpleNamespace(reward_return=None, reward_var=None, reward_count=0.0)
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    batch = {
        ("next", "reward"): rewards.clone(),
        ("next", "done"): torch.tensor([[0.0], [0.0], [1.0]]),
    }

    normalize_rewards_for_training(config, state, batch, device=torch.device("cpu"))

    assert state.reward_count == 3.0
    assert state.reward_var is not None
    assert state.reward_return is not None
    assert not torch.equal(batch[("next", "reward")], rewards)


def test_normalize_rewards_for_training_noops_when_disabled() -> None:
    config = SimpleNamespace(env=SimpleNamespace(normalize_reward=False))
    state = SimpleNamespace()
    rewards = torch.tensor([[1.0]])
    batch = {("next", "reward"): rewards.clone()}

    normalize_rewards_for_training(config, state, batch, device=torch.device("cpu"))

    torch.testing.assert_close(batch[("next", "reward")], rewards)
