from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from optimizer.designer_errors import NoSuchDesignerError
from optimizer.ppo_common import normalize_advantages
from optimizer.ppo_designer import (
    PPOPGConfig,
    PPOPGDesigner,
    compute_episode_return_advantages,
    merge_trajectories,
)
from optimizer.trajectory import Trajectory
from policies.actor_critic_mlp_policy import ActorCriticMLPPolicy
from policies.actor_mlp_policy import ActorMLPPolicy


def _env_conf(obs_dim: int = 4, act_dim: int = 2):
    return SimpleNamespace(
        problem_seed=0,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(obs_dim,))),
        action_space=SimpleNamespace(shape=(act_dim,)),
    )


def _make_actor_policy(obs_dim: int = 4, act_dim: int = 2):
    p = ActorMLPPolicy(_env_conf(obs_dim, act_dim), (8,))
    p.eval()
    return p


def _make_trajectory(num_steps: int = 10, obs_dim: int = 4, act_dim: int = 2, *, with_values: bool):
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
        values=np.ones(num_steps, dtype=np.float32) if with_values else None,
        dones=dones,
    )


def test_ppopg_config_defaults():
    cfg = PPOPGConfig()
    assert cfg.lr == 3e-4
    assert cfg.clip_coef == 0.2
    assert cfg.epochs == 4
    assert cfg.gamma == 1.0
    assert not hasattr(cfg, "vf_coef")


def test_compute_episode_return_advantages_single_episode():
    rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dones = np.array([False, False, True], dtype=bool)
    adv = compute_episode_return_advantages(rewards, dones)
    np.testing.assert_allclose(adv, 6.0)


def test_compute_episode_return_advantages_two_segments():
    rewards = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    dones = np.array([False, True, False, True], dtype=bool)
    adv = compute_episode_return_advantages(rewards, dones)
    np.testing.assert_allclose(adv[:2], 2.0)
    np.testing.assert_allclose(adv[2:], 4.0)


def test_compute_episode_return_advantages_discounted():
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    dones = np.array([False, True], dtype=bool)
    adv = compute_episode_return_advantages(rewards, dones, gamma=0.5)
    np.testing.assert_allclose(adv, 1.0 + 0.5 * 1.0)


def test_normalize_advantages_single_timestep_is_finite():
    adv = normalize_advantages(np.array([3.0], dtype=np.float32), torch.device("cpu"))
    assert adv.shape == (1,)
    assert torch.isfinite(adv).all()
    torch.testing.assert_close(adv, torch.zeros(1))


def test_normalize_advantages_constant_episode_return_vector_not_zeroed():
    adv = normalize_advantages(np.full(5, 15.0, dtype=np.float32), torch.device("cpu"))
    torch.testing.assert_close(adv, torch.full((5,), 15.0))


@patch("optimizer.ppo_pg_designer.collect_trajectory")
def test_ppopg_single_episode_updates_policy_with_ent_coef_zero(mock_collect):
    np.random.seed(0)
    num_steps = 5
    mock_collect.return_value = Trajectory(
        rreturn=15.0,
        states=np.random.randn(4, num_steps).astype(np.float32),
        actions=np.clip(np.random.randn(2, num_steps).astype(np.float32) * 0.3, -0.99, 0.99),
        num_steps=num_steps,
        rewards=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        log_probs=-np.random.randn(num_steps).astype(np.float32),
        values=None,
        dones=np.array([False, False, False, False, True], dtype=bool),
    )

    policy = _make_actor_policy()
    policy.train()
    original_params = policy.get_params().copy()
    designer = PPOPGDesigner(policy, _env_conf(), epochs=2, ent_coef=0.0)

    result = designer([], num_arms=1)

    assert not np.allclose(original_params, result[0].get_params())


def test_merge_trajectories_value_less():
    traj1 = _make_trajectory(5, with_values=False)
    traj2 = _make_trajectory(7, with_values=False)
    merged = merge_trajectories([traj1, traj2])
    assert merged.values is None
    assert merged.rewards.shape == (12,)


def test_merge_trajectories_empty_raises():
    with pytest.raises(NoSuchDesignerError, match="at least one trajectory"):
        merge_trajectories([])


def test_ppopg_designer_num_arms_zero_raises():
    policy = _make_actor_policy()
    designer = PPOPGDesigner(policy, _env_conf())
    with pytest.raises(NoSuchDesignerError, match="num_arms >= 1"):
        designer([], num_arms=0)


def test_merge_trajectories_mixed_values_raises():
    traj1 = _make_trajectory(5, with_values=True)
    traj2 = _make_trajectory(5, with_values=False)
    with pytest.raises(NoSuchDesignerError, match="mixed"):
        merge_trajectories([traj1, traj2])


def _make_realistic_trajectory(num_steps: int, seed: int, *, with_values: bool):
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
        values=np.full(num_steps, 0.5, dtype=np.float32) if with_values else None,
        dones=dones,
    )


def test_merged_episode_return_matches_separate_per_arm():
    traj1 = _make_realistic_trajectory(5, seed=42, with_values=False)
    traj2 = _make_realistic_trajectory(7, seed=43, with_values=False)
    adv1 = compute_episode_return_advantages(traj1.rewards, traj1.dones)
    adv2 = compute_episode_return_advantages(traj2.rewards, traj2.dones)
    merged = merge_trajectories([traj1, traj2])
    merged_adv = compute_episode_return_advantages(merged.rewards, merged.dones)
    np.testing.assert_allclose(merged_adv, np.concatenate([adv1, adv2]))


def test_merged_episode_return_advantages_split_per_arm_without_terminal_done():
    traj1 = _make_trajectory(3, with_values=False)
    traj1.dones[-1] = False
    traj2 = _make_trajectory(3, with_values=False)
    merged = merge_trajectories([traj1, traj2])
    adv = compute_episode_return_advantages(merged.rewards, merged.dones)
    np.testing.assert_allclose(adv[:3], 3.0)
    np.testing.assert_allclose(adv[3:], 3.0)


def test_ppopg_designer_rejects_actor_critic():
    policy = ActorCriticMLPPolicy(_env_conf(), (8,))
    designer = PPOPGDesigner(policy, _env_conf())
    with pytest.raises(NoSuchDesignerError, match="actor-only"):
        designer([], num_arms=1)


@pytest.mark.parametrize("num_arms", [1, 3])
@patch("optimizer.ppo_pg_designer.collect_trajectory")
def test_ppopg_designer_call_returns_updated_policy(mock_collect, num_arms):
    mock_collect.return_value = _make_trajectory(5, with_values=False)

    policy = _make_actor_policy()
    designer = PPOPGDesigner(policy, _env_conf(), epochs=1)

    result = designer([], num_arms=num_arms)

    assert len(result) == 1
    assert isinstance(result[0], ActorMLPPolicy)
    assert mock_collect.call_count == num_arms


@pytest.mark.parametrize(
    "missing_field",
    ["log_probs", "rewards", "dones"],
)
@patch("optimizer.ppo_pg_designer.collect_trajectory")
def test_ppopg_designer_raises_if_trajectory_field_missing(mock_collect, missing_field):
    field_values = {
        "rewards": np.ones(10, dtype=np.float32),
        "log_probs": -np.ones(10, dtype=np.float32),
        "dones": np.zeros(10, dtype=bool),
    }
    field_values[missing_field] = None

    traj = Trajectory(
        rreturn=100.0,
        states=np.random.randn(4, 10).astype(np.float32),
        actions=np.random.randn(2, 10).astype(np.float32),
        num_steps=10,
        values=None,
        **field_values,
    )
    mock_collect.return_value = traj

    policy = _make_actor_policy()
    designer = PPOPGDesigner(policy, _env_conf(), epochs=1)

    with pytest.raises(NoSuchDesignerError, match=missing_field):
        designer([], num_arms=1)


@patch("optimizer.ppo_pg_designer.collect_trajectory")
def test_ppopg_designer_uses_parent_policy_from_data(mock_collect):
    rollout_params = []

    def _capture_rollout(env, pol):
        rollout_params.append(pol.get_params().copy())
        return _make_trajectory(num_steps=5, with_values=False)

    mock_collect.side_effect = _capture_rollout

    policy = _make_actor_policy()
    parent_policy = _make_actor_policy()
    env_conf = _env_conf()
    designer = PPOPGDesigner(policy, env_conf, epochs=1)

    datum = SimpleNamespace(policy=parent_policy)
    result = designer([datum], num_arms=1)

    assert len(result) == 1
    mock_collect.assert_called_once()
    assert len(rollout_params) == 1
    np.testing.assert_allclose(rollout_params[0], parent_policy.get_params())
    assert result[0] is mock_collect.call_args[0][1]


@patch("optimizer.ppo_pg_designer.collect_trajectory")
def test_ppopg_designer_modifies_policy_params(mock_collect):
    mock_collect.return_value = _make_trajectory(num_steps=5, with_values=False)

    policy = _make_actor_policy()
    original_params = policy.get_params().copy()
    designer = PPOPGDesigner(policy, _env_conf(), epochs=2)

    result = designer([], num_arms=1)
    new_params = result[0].get_params()

    assert not np.allclose(original_params, new_params)


@patch("optimizer.ppo_pg_designer.collect_trajectory")
def test_ppopg_designer_returned_policy_is_cloneable(mock_collect):
    mock_collect.return_value = _make_trajectory(num_steps=5, with_values=False)

    policy = _make_actor_policy()
    designer = PPOPGDesigner(policy, _env_conf(), epochs=1)

    result = designer([], num_arms=1)

    cloned = result[0].clone()
    np.testing.assert_allclose(cloned.get_params(), result[0].get_params())


@patch("optimizer.ppo_pg_designer.collect_trajectory")
def test_ppopg_designer_telemetry_rollout_fields(mock_collect):
    mock_collect.return_value = _make_trajectory(5, with_values=False)

    policy = _make_actor_policy()
    designer = PPOPGDesigner(policy, _env_conf(), epochs=1)

    telemetry = MagicMock()
    designer([], num_arms=2, telemetry=telemetry)

    telemetry.set_num_rollout_workers.assert_called_once_with(2)
    telemetry.set_dt_rollout.assert_called_once()
    telemetry.set_dt_fit.assert_called_once()
    telemetry.set_dt_select.assert_called_once_with(0.0)
