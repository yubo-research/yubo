from __future__ import annotations

import torch
import torch.nn as nn

# Target clipping to prevent Q divergence from unbounded targets (SAC and GAC).
_TARGET_CLIP = 1e6


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
    # Clamp log_prob to prevent target explosion when policy assigns very low prob.
    next_log_prob = next_log_prob.clamp(min=-20.0)
    q_target = torch.min(q1_target, q2_target) - alpha * next_log_prob
    target = rew + float(gamma) * (1.0 - done) * q_target
    return target.clamp(-_TARGET_CLIP, _TARGET_CLIP)


def sac_critic_loss(q1: torch.Tensor, q2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)


def sac_actor_loss(alpha: torch.Tensor | float, log_prob: torch.Tensor, q_pi: torch.Tensor) -> torch.Tensor:
    return (alpha * log_prob - q_pi).mean()


def sac_alpha_loss(log_alpha: torch.Tensor, log_prob: torch.Tensor, *, target_entropy: float) -> torch.Tensor:
    return -(log_alpha * (log_prob + float(target_entropy)).detach()).mean()


def compute_gac_target(
    rew: torch.Tensor,
    done: torch.Tensor,
    *,
    gamma: float,
    q1_target: torch.Tensor,
    q2_target: torch.Tensor,
    next_kappa: torch.Tensor,
) -> torch.Tensor:
    """GAC target: y = r + γ(1-done)(min(Q') - κ'). No alpha*log_prob."""
    next_kappa = next_kappa.clamp(min=-2.0, max=5.0)  # Paper: κ ∈ [0,5] empirically
    q_target = torch.min(q1_target, q2_target) - next_kappa
    target = rew + float(gamma) * (1.0 - done) * q_target
    return target.clamp(-_TARGET_CLIP, _TARGET_CLIP)


def gac_actor_loss(kappa: torch.Tensor, q_pi: torch.Tensor) -> torch.Tensor:
    """GAC actor loss: minimize (κ - min(Q)) to maximize Q - κ (soft value V = Q - κ)."""
    kappa_c = kappa.clamp(min=-2.0, max=5.0)  # Prevent gradient explosion from unbounded κ
    return (kappa_c - q_pi).mean()


def soft_update_module(target: nn.Module, source: nn.Module, *, tau: float) -> None:
    tau_f = float(tau)
    one_minus_tau = 1.0 - tau_f
    for p_tgt, p_src in zip(target.parameters(), source.parameters(), strict=True):
        p_tgt.data.mul_(one_minus_tau).add_(tau_f * p_src.data)
