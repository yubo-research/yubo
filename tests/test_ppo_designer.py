from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from optimizer.designer_errors import NoSuchDesignerError
from optimizer.ppo_designer import PPOConfig, PPODesigner, compute_gae, merge_trajectories
from optimizer.trajectory import Trajectory
from policies.actor_critic_mlp_policy import ActorCriticMLPPolicy


def _env_conf(obs_dim: int = 4, act_dim: int = 2):
    return SimpleNamespace(
        problem_seed=0,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(obs_dim,))),
        action_space=SimpleNamespace(shape=(act_dim,)),
    )


def _make_policy(obs_dim: int = 4, act_dim: int = 2):
    env_conf = _env_conf(obs_dim, act_dim)
    p = ActorCriticMLPPolicy(env_conf, (8,))
    p.eval()
    return p


def _make_trajectory(num_steps: int = 10, obs_dim: int = 4, act_dim: int = 2):
    """Create a fake trajectory with PPO data."""
    np.random.seed(42)
    dones = np.zeros(num_steps, dtype=bool)
    dones[-1] = True
    return Trajectory(
        rreturn=100.0,
        states=np.random.randn(obs_dim, num_steps).astype(np.float32) * 0.1,
        actions=np.clip(np.random.randn(act_dim, num_steps).astype(np.float32) * 0.3, -0.99, 0.99),
        num_steps=num_steps,
        rewards=np.ones(num_steps, dtype=np.float32),
        log_probs=-np.ones(num_steps, dtype=np.float32),
        values=np.ones(num_steps, dtype=np.float32),
        dones=dones,
    )


def test_ppo_config_defaults():
    cfg = PPOConfig()
    assert cfg.lr == 3e-4
    assert cfg.clip_coef == 0.2
    assert cfg.epochs == 4
    assert cfg.gamma == 0.99
    assert cfg.gae_lambda == 0.95
    assert cfg.vf_coef == 0.5
    assert cfg.ent_coef == 0.01
    assert cfg.max_grad_norm == 0.5


def test_compute_gae_shape():
    rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    values = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    dones = np.array([False, False, True], dtype=bool)

    advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)

    assert advantages.shape == (3,)
    assert returns.shape == (3,)
    assert returns.dtype == np.float32


def test_compute_gae_episode_terminated():
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dones = np.array([False, False, True], dtype=bool)

    advantages, returns = compute_gae(rewards, values, dones, gamma=1.0, gae_lambda=1.0)

    np.testing.assert_allclose(advantages[2], 1.0, atol=1e-5)
    np.testing.assert_allclose(returns, advantages, atol=1e-5)


def test_compute_gae_with_bootstrap():
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.5], dtype=np.float32)
    dones = np.array([False, False], dtype=bool)

    adv1, _ = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95, last_value=0.0)
    adv2, _ = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95, last_value=1.0)

    assert not np.allclose(adv1, adv2)


def test_ppo_designer_init():
    policy = _make_policy()
    env_conf = _env_conf()
    designer = PPODesigner(policy, env_conf, lr=1e-3, epochs=2)

    assert designer._config.lr == 1e-3
    assert designer._config.epochs == 2


def test_ppo_designer_num_arms_zero_raises():
    """PPODesigner requires num_arms >= 1."""
    policy = _make_policy()
    env_conf = _env_conf()
    designer = PPODesigner(policy, env_conf)

    with pytest.raises(NoSuchDesignerError, match="num_arms >= 1"):
        designer([], num_arms=0)


def test_ppo_designer_validates_policy_missing_methods():
    class BadPolicy:
        def clone(self):
            return self

    designer = PPODesigner(BadPolicy(), _env_conf())
    with pytest.raises(NoSuchDesignerError, match="ActorCriticMLPPolicy"):
        designer([], num_arms=1)


def test_ppo_designer_init_with_config():
    policy = _make_policy()
    env_conf = _env_conf()
    cfg = PPOConfig(lr=1e-2, epochs=8)
    designer = PPODesigner(policy, env_conf, config=cfg)

    assert designer._config.lr == 1e-2
    assert designer._config.epochs == 8
    assert designer._config is cfg


@patch("optimizer.ppo_designer.collect_trajectory")
def test_ppo_designer_call_returns_updated_policy(mock_collect):
    mock_collect.return_value = _make_trajectory(num_steps=5)

    policy = _make_policy()
    env_conf = _env_conf()
    designer = PPODesigner(policy, env_conf, epochs=1)

    result = designer([], num_arms=1)

    assert len(result) == 1
    assert isinstance(result[0], ActorCriticMLPPolicy)
    mock_collect.assert_called_once()


@patch("optimizer.ppo_designer.collect_trajectory")
def test_ppo_designer_uses_parent_policy_from_data(mock_collect):
    mock_collect.return_value = _make_trajectory(num_steps=5)

    policy = _make_policy()
    parent_policy = _make_policy()
    env_conf = _env_conf()
    designer = PPODesigner(policy, env_conf, epochs=1)

    datum = SimpleNamespace(policy=parent_policy)
    result = designer([datum], num_arms=1)

    assert len(result) == 1


@patch("optimizer.ppo_designer.collect_trajectory")
def test_ppo_designer_with_telemetry(mock_collect):
    mock_collect.return_value = _make_trajectory(num_steps=5)

    policy = _make_policy()
    env_conf = _env_conf()
    designer = PPODesigner(policy, env_conf, epochs=1)

    telemetry = MagicMock()
    designer([], num_arms=1, telemetry=telemetry)

    telemetry.set_dt_fit.assert_called_once()
    telemetry.set_dt_select.assert_called_once()


@patch("optimizer.ppo_designer.collect_trajectory")
def test_ppo_designer_modifies_policy_params(mock_collect):
    mock_collect.return_value = _make_trajectory(num_steps=5)

    policy = _make_policy()
    original_params = policy.get_params().copy()
    env_conf = _env_conf()
    designer = PPODesigner(policy, env_conf, epochs=2)

    result = designer([], num_arms=1)
    new_params = result[0].get_params()

    assert not np.allclose(original_params, new_params)


@pytest.mark.parametrize(
    "missing_field",
    ["log_probs", "values", "rewards", "dones"],
)
@patch("optimizer.ppo_designer.collect_trajectory")
def test_ppo_designer_raises_if_trajectory_field_missing(mock_collect, missing_field):
    """PPODesigner validates all required trajectory fields."""
    field_values = {
        "rewards": np.ones(10, dtype=np.float32),
        "log_probs": -np.ones(10, dtype=np.float32),
        "values": np.ones(10, dtype=np.float32),
        "dones": np.zeros(10, dtype=bool),
    }
    field_values[missing_field] = None

    traj = Trajectory(
        rreturn=100.0,
        states=np.random.randn(4, 10).astype(np.float32),
        actions=np.random.randn(2, 10).astype(np.float32),
        num_steps=10,
        **field_values,
    )
    mock_collect.return_value = traj

    policy = _make_policy()
    designer = PPODesigner(policy, _env_conf(), epochs=1)

    with pytest.raises(NoSuchDesignerError, match=missing_field):
        designer([], num_arms=1)


@patch("optimizer.ppo_designer.collect_trajectory")
def test_ppo_designer_validates_get_action_and_value(mock_collect):
    """Verify PPODesigner validates get_action_and_value method exists.

    This test ensures that a policy without get_action_and_value() is rejected
    at validation time, rather than causing an AttributeError in _ppo_update_epoch.
    """
    import torch.nn as nn

    mock_collect.return_value = _make_trajectory(num_steps=5)

    class PolicyWithLastButNoGetActionAndValue(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def clone(self):
            return PolicyWithLastButNoGetActionAndValue()

        def last_log_probs(self):
            return None

        def last_values(self):
            return None

    policy = PolicyWithLastButNoGetActionAndValue()
    designer = PPODesigner(policy, _env_conf(), epochs=1)

    with pytest.raises(NoSuchDesignerError, match="get_action_and_value"):
        designer([], num_arms=1)


def test_merge_trajectories_single():
    """merge_trajectories returns the single trajectory unchanged."""
    traj = _make_trajectory(num_steps=5)
    merged = merge_trajectories([traj])
    assert merged is traj


def test_merge_trajectories_multiple():
    """merge_trajectories correctly concatenates multiple trajectories."""
    traj1 = _make_trajectory(num_steps=5, obs_dim=4, act_dim=2)
    traj2 = _make_trajectory(num_steps=7, obs_dim=4, act_dim=2)
    merged = merge_trajectories([traj1, traj2])

    assert merged.states.shape == (4, 12)
    assert merged.actions.shape == (2, 12)
    assert merged.rewards.shape == (12,)
    assert merged.log_probs.shape == (12,)
    assert merged.values.shape == (12,)
    assert merged.dones.shape == (12,)
    assert merged.num_steps == 12


def _make_realistic_trajectory(num_steps: int, seed: int):
    """Create a trajectory with done=True at last step (like real collection)."""
    np.random.seed(seed)
    dones = np.zeros(num_steps, dtype=bool)
    dones[-1] = True
    return Trajectory(
        rreturn=float(num_steps),
        states=np.random.randn(4, num_steps).astype(np.float32),
        actions=np.clip(np.random.randn(2, num_steps).astype(np.float32) * 0.3, -0.99, 0.99),
        num_steps=num_steps,
        rewards=np.ones(num_steps, dtype=np.float32),
        log_probs=-np.ones(num_steps, dtype=np.float32),
        values=np.full(num_steps, 0.5, dtype=np.float32),
        dones=dones,
    )


def test_merged_gae_matches_separate_gae():
    """GAE on merged trajectories should match per-trajectory GAE when done flags are set."""
    traj1 = _make_realistic_trajectory(num_steps=5, seed=42)
    traj2 = _make_realistic_trajectory(num_steps=7, seed=43)

    adv1, ret1 = compute_gae(traj1.rewards, traj1.values, traj1.dones, gamma=0.99, gae_lambda=0.95)
    adv2, ret2 = compute_gae(traj2.rewards, traj2.values, traj2.dones, gamma=0.99, gae_lambda=0.95)

    merged = merge_trajectories([traj1, traj2])
    merged_adv, merged_ret = compute_gae(merged.rewards, merged.values, merged.dones, gamma=0.99, gae_lambda=0.95)

    expected_adv = np.concatenate([adv1, adv2])
    expected_ret = np.concatenate([ret1, ret2])

    np.testing.assert_allclose(merged_adv, expected_adv, rtol=1e-5)
    np.testing.assert_allclose(merged_ret, expected_ret, rtol=1e-5)


def test_make_trajectory_helper_has_realistic_dones():
    """Verify that _make_trajectory has done=True at last step like real trajectories.

    Real trajectories from collect_trajectory have done=True at the last step.
    Without this, GAE computed on merged trajectories incorrectly bleeds
    advantages across trajectory boundaries.
    """
    traj = _make_trajectory(num_steps=5)

    assert traj.dones[-1] is True or traj.dones[-1] == 1, "_make_trajectory should set done=True at last step to match real collection"


@patch("optimizer.ppo_designer.collect_trajectory")
def test_ppo_designer_num_arms_multiple(mock_collect):
    """PPODesigner accepts num_arms > 1 and returns single updated policy."""
    mock_collect.return_value = _make_trajectory(num_steps=5)

    policy = _make_policy()
    env_conf = _env_conf()
    designer = PPODesigner(policy, env_conf, epochs=1)

    result = designer([], num_arms=3)

    assert len(result) == 1
    assert isinstance(result[0], ActorCriticMLPPolicy)
    assert mock_collect.call_count == 3


@patch("optimizer.ppo_designer.collect_trajectory")
def test_ppo_designer_telemetry_rollout_fields(mock_collect):
    """PPODesigner sets rollout telemetry fields when available."""
    mock_collect.return_value = _make_trajectory(num_steps=5)

    policy = _make_policy()
    env_conf = _env_conf()
    designer = PPODesigner(policy, env_conf, epochs=1)

    telemetry = MagicMock()
    telemetry.set_num_rollout_workers = MagicMock()
    telemetry.set_dt_rollout = MagicMock()

    designer([], num_arms=2, telemetry=telemetry)

    telemetry.set_num_rollout_workers.assert_called_once_with(2)
    telemetry.set_dt_rollout.assert_called_once()
