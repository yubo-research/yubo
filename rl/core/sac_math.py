"""Backend-agnostic SAC math helpers."""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_sac_target(
    rew: torch.Tensor,
    done: torch.Tensor,
    *,
    gamma: float,
    q1_target: torch.Tensor,
    q2_target: torch.Tensor,
    alpha: torch.Tensor | float,
    next_log_prob: torch.Tensor,
) -> torch.Tensor:
    q_target = torch.min(q1_target, q2_target) - alpha * next_log_prob
    return rew + float(gamma) * (1.0 - done) * q_target


def sac_critic_loss(q1: torch.Tensor, q2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)


def sac_actor_loss(alpha: torch.Tensor | float, log_prob: torch.Tensor, q_pi: torch.Tensor) -> torch.Tensor:
    return (alpha * log_prob - q_pi).mean()


def sac_alpha_loss(log_alpha: torch.Tensor, log_prob: torch.Tensor, *, target_entropy: float) -> torch.Tensor:
    return -(log_alpha * (log_prob + float(target_entropy)).detach()).mean()


def soft_update_module(target: nn.Module, source: nn.Module, *, tau: float) -> None:
    tau_f = float(tau)
    one_minus_tau = 1.0 - tau_f
    for p_tgt, p_src in zip(target.parameters(), source.parameters(), strict=True):
        p_tgt.data.mul_(one_minus_tau).add_(tau_f * p_src.data)
