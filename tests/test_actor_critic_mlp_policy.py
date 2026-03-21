from types import SimpleNamespace

import numpy as np
import torch

from policies.actor_critic_mlp_policy import ActorCriticMLPPolicy, ActorCriticMLPPolicyFactory


def _env_conf(obs_dim: int = 4, act_dim: int = 2):
    return SimpleNamespace(
        problem_seed=0,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(obs_dim,))),
        action_space=SimpleNamespace(shape=(act_dim,)),
    )


def test_actor_critic_mlp_policy_factory_builds_policy():
    fac = ActorCriticMLPPolicyFactory((8, 8), share_backbone=False, log_std_init=-0.5)
    p = fac(_env_conf())
    assert isinstance(p, ActorCriticMLPPolicy)
    assert p.actor_backbone is not p.critic_backbone
    assert float(p.log_std.data.mean()) == -0.5


def test_actor_critic_mlp_policy_shared_backbone():
    p = ActorCriticMLPPolicy(_env_conf(), (8,), share_backbone=True)
    assert p.actor_backbone is p.critic_backbone


def test_actor_critic_mlp_policy_forward_get_value_get_action_and_value():
    p = ActorCriticMLPPolicy(_env_conf(), (16,))
    p.train()
    obs = torch.randn(3, 4)
    act_det = p.forward(obs)
    assert act_det.shape == (3, 2)
    assert act_det.abs().max() <= 1.0 + 1e-5

    v = p.get_value(obs)
    assert v.shape == (3,)

    a, lp, ent, val = p.get_action_and_value(obs)
    assert a.shape == (3, 2)
    assert lp.shape == (3,)
    assert ent.shape == (3,)
    assert val.shape == (3,)
    assert p.last_log_probs() is lp
    assert p.last_values() is val

    a2, lp2, ent2, val2 = p.get_action_and_value(obs, action=a.detach())
    assert torch.allclose(a2, a)
    assert lp2.shape == (3,)
    assert ent2.shape == (3,)
    assert val2.shape == (3,)


def test_actor_critic_mlp_policy_numpy_call():
    p = ActorCriticMLPPolicy(_env_conf(), (8,))
    p.eval()
    out = p(np.zeros(4, dtype=np.float32))
    assert out.shape == (2,)


def test_actor_critic_mlp_policy_missing_spaces_raises():
    bad = SimpleNamespace(problem_seed=0)
    try:
        ActorCriticMLPPolicy(bad, (8,))
    except ValueError as e:
        assert "ensure_spaces" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_actor_critic_mlp_policy_numpy_call_populates_log_probs_and_values():
    """Regression test: __call__ must populate last_log_probs/last_values.

    Ensures collect_trajectory gets valid log_probs/values when used with
    ActorCriticMLPPolicy (previously returned nan before the fix).
    """
    p = ActorCriticMLPPolicy(_env_conf(), (8,))
    p.eval()

    state = np.zeros(4, dtype=np.float32)
    _action = p(state)

    log_probs = p.last_log_probs()
    values = p.last_values()

    assert log_probs is not None, "last_log_probs() should not be None after __call__"
    assert values is not None, "last_values() should not be None after __call__"
    assert not np.isnan(log_probs.detach().cpu().numpy()).any(), "log_probs should not be nan"
    assert not np.isnan(values.detach().cpu().numpy()).any(), "values should not be nan"
