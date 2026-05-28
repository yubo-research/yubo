from types import SimpleNamespace

import numpy as np
import torch

from policies.actor_mlp_policy import ActorMLPPolicy


def _env_conf(obs_dim: int = 4, act_dim: int = 2):
    return SimpleNamespace(
        problem_seed=0,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(obs_dim,))),
        action_space=SimpleNamespace(shape=(act_dim,)),
    )


def test_actor_mlp_call_numpy_action():
    policy = ActorMLPPolicy(_env_conf(), (8,))
    policy.eval()
    action = policy(np.zeros(4, dtype=np.float32))
    assert action.shape == (2,)


def test_actor_mlp_last_log_probs_after_call():
    policy = ActorMLPPolicy(_env_conf(), (8,))
    policy.eval()
    policy(np.zeros(4, dtype=np.float32))
    lp = policy.last_log_probs()
    assert lp is not None
    assert lp.shape == ()


def test_actor_mlp_log_prob_matches_distribution():
    policy = ActorMLPPolicy(_env_conf(), (8,))
    policy.eval()
    obs = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    action, log_prob, _entropy = policy.get_action_and_value(obs, action=obs.new_zeros(1, 2))
    dist = policy._distribution(obs)
    u = torch.atanh(torch.clamp(action, -0.999, 0.999))
    expected = dist.log_prob(u) - torch.log(1.0 - action.pow(2) + 1e-06)
    expected = expected.sum(-1)
    torch.testing.assert_close(log_prob, expected)


def test_actor_mlp_last_log_probs_shape():
    policy = ActorMLPPolicy(_env_conf(), (8,))
    policy.eval()
    obs = torch.randn(3, 4)
    policy.get_action_and_value(obs)
    lp = policy.last_log_probs()
    assert lp is not None
    assert lp.shape == (3,)


def test_actor_mlp_no_last_values():
    policy = ActorMLPPolicy(_env_conf(), (8,))
    assert not hasattr(policy, "last_values")


def test_actor_mlp_clone_and_params():
    policy = ActorMLPPolicy(_env_conf(), (8,))
    policy.eval()
    p0 = policy.get_params().copy()
    cloned = policy.clone()
    assert cloned is not policy
    np.testing.assert_allclose(cloned.get_params(), p0)
    delta = np.full_like(p0, 0.25)
    cloned.set_params(delta)
    np.testing.assert_allclose(cloned.get_params(), delta, rtol=1e-5)
    np.testing.assert_allclose(policy.get_params(), p0)
