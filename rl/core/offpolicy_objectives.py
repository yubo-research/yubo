from __future__ import annotations

import torch
import torch.nn as nn


def entropy_regularized_target(
    reward: torch.Tensor,
    done: torch.Tensor,
    *,
    gamma: float,
    next_q1: torch.Tensor,
    next_q2: torch.Tensor,
    entropy_temperature: torch.Tensor | float,
    next_log_prob: torch.Tensor,
) -> torch.Tensor:
    next_value = torch.min(next_q1, next_q2) - entropy_temperature * next_log_prob
    return reward + float(gamma) * (1.0 - done) * next_value


def twin_critic_mse_loss(q1: torch.Tensor, q2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)


def entropy_regularized_policy_loss(
    entropy_temperature: torch.Tensor | float,
    log_prob: torch.Tensor,
    q_value: torch.Tensor,
) -> torch.Tensor:
    return (entropy_temperature * log_prob - q_value).mean()


def temperature_objective(log_temperature: torch.Tensor, log_prob: torch.Tensor, *, target_entropy: float) -> torch.Tensor:
    return -(log_temperature * (log_prob + float(target_entropy)).detach()).mean()


def polyak_update_parameters(target: nn.Module, source: nn.Module, *, tau: float) -> None:
    tau_f = float(tau)
    one_minus_tau = 1.0 - tau_f
    for target_parameter, source_parameter in zip(target.parameters(), source.parameters(), strict=True):
        target_parameter.data.mul_(one_minus_tau).add_(tau_f * source_parameter.data)
