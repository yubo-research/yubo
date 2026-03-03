import numpy as np
import torch
import torch.nn as nn

from rl.core.offpolicy_objectives import (
    entropy_regularized_policy_loss,
    entropy_regularized_target,
    polyak_update_parameters,
    temperature_objective,
    twin_critic_mse_loss,
)


def test_entropy_regularized_target_matches_manual_formula():
    rew = torch.as_tensor([1.0, 2.0], dtype=torch.float32)
    done = torch.as_tensor([0.0, 1.0], dtype=torch.float32)
    q1_t = torch.as_tensor([3.0, 5.0], dtype=torch.float32)
    q2_t = torch.as_tensor([4.0, 4.0], dtype=torch.float32)
    log_prob = torch.as_tensor([0.5, 0.25], dtype=torch.float32)
    alpha = torch.as_tensor(0.2, dtype=torch.float32)

    target = entropy_regularized_target(
        rew,
        done,
        gamma=0.99,
        next_q1=q1_t,
        next_q2=q2_t,
        entropy_temperature=alpha,
        next_log_prob=log_prob,
    )

    manual = rew + 0.99 * (1.0 - done) * (torch.min(q1_t, q2_t) - alpha * log_prob)
    assert torch.allclose(target, manual)


def test_offpolicy_loss_helpers_match_expected():
    q1 = torch.as_tensor([1.0, 2.0], dtype=torch.float32)
    q2 = torch.as_tensor([2.0, 3.0], dtype=torch.float32)
    target = torch.as_tensor([0.0, 2.0], dtype=torch.float32)
    critic = twin_critic_mse_loss(q1, q2, target)
    expected_critic = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
    assert torch.allclose(critic, expected_critic)

    alpha = torch.as_tensor(0.1, dtype=torch.float32)
    log_prob = torch.as_tensor([0.3, -0.2], dtype=torch.float32)
    q_pi = torch.as_tensor([1.0, 0.5], dtype=torch.float32)
    actor = entropy_regularized_policy_loss(alpha, log_prob, q_pi)
    expected_actor = (alpha * log_prob - q_pi).mean()
    assert torch.allclose(actor, expected_actor)

    log_alpha = torch.tensor(np.log(0.2), dtype=torch.float32, requires_grad=True)
    alpha_l = temperature_objective(log_alpha, log_prob, target_entropy=-1.0)
    expected_alpha = -(log_alpha * (log_prob - 1.0).detach()).mean()
    assert torch.allclose(alpha_l, expected_alpha)


def test_polyak_update_parameters_interpolates_params():
    src = nn.Linear(2, 2, bias=False)
    tgt = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        src.weight.fill_(2.0)
        tgt.weight.fill_(0.0)
    polyak_update_parameters(tgt, src, tau=0.25)
    expected = torch.full_like(tgt.weight, 0.5)
    assert torch.allclose(tgt.weight, expected)
