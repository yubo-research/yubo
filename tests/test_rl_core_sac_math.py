import numpy as np
import torch
import torch.nn as nn

from rl.core.sac_math import (
    compute_sac_target,
    sac_actor_loss,
    sac_alpha_loss,
    sac_critic_loss,
    soft_update_module,
)


def test_compute_sac_target_matches_manual_formula():
    rew = torch.as_tensor([1.0, 2.0], dtype=torch.float32)
    done = torch.as_tensor([0.0, 1.0], dtype=torch.float32)
    q1_t = torch.as_tensor([3.0, 5.0], dtype=torch.float32)
    q2_t = torch.as_tensor([4.0, 4.0], dtype=torch.float32)
    log_prob = torch.as_tensor([0.5, 0.25], dtype=torch.float32)
    alpha = torch.as_tensor(0.2, dtype=torch.float32)

    target = compute_sac_target(
        rew,
        done,
        gamma=0.99,
        q1_target=q1_t,
        q2_target=q2_t,
        alpha=alpha,
        next_log_prob=log_prob,
    )

    manual = rew + 0.99 * (1.0 - done) * (torch.min(q1_t, q2_t) - alpha * log_prob)
    assert torch.allclose(target, manual)


def test_sac_loss_helpers_match_expected():
    q1 = torch.as_tensor([1.0, 2.0], dtype=torch.float32)
    q2 = torch.as_tensor([2.0, 3.0], dtype=torch.float32)
    target = torch.as_tensor([0.0, 2.0], dtype=torch.float32)
    critic = sac_critic_loss(q1, q2, target)
    expected_critic = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
    assert torch.allclose(critic, expected_critic)

    alpha = torch.as_tensor(0.1, dtype=torch.float32)
    log_prob = torch.as_tensor([0.3, -0.2], dtype=torch.float32)
    q_pi = torch.as_tensor([1.0, 0.5], dtype=torch.float32)
    actor = sac_actor_loss(alpha, log_prob, q_pi)
    expected_actor = (alpha * log_prob - q_pi).mean()
    assert torch.allclose(actor, expected_actor)

    log_alpha = torch.tensor(np.log(0.2), dtype=torch.float32, requires_grad=True)
    alpha_l = sac_alpha_loss(log_alpha, log_prob, target_entropy=-1.0)
    expected_alpha = -(log_alpha * (log_prob - 1.0).detach()).mean()
    assert torch.allclose(alpha_l, expected_alpha)


def test_soft_update_module_interpolates_params():
    src = nn.Linear(2, 2, bias=False)
    tgt = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        src.weight.fill_(2.0)
        tgt.weight.fill_(0.0)
    soft_update_module(tgt, src, tau=0.25)
    expected = torch.full_like(tgt.weight, 0.5)
    assert torch.allclose(tgt.weight, expected)
