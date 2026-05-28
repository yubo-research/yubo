import torch
from torch.distributions import Normal


def atanh(x: torch.Tensor, eps: float = 1e-06) -> torch.Tensor:
    """Inverse hyperbolic tangent with clamping for numerical stability."""
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def tanh_gaussian_action_log_prob_entropy(
    dist: Normal,
    action: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if action is None:
        u = dist.rsample()
        action = torch.tanh(u)
    else:
        u = atanh(action)
    log_prob = dist.log_prob(u) - torch.log(1.0 - action.pow(2) + 1e-06)
    log_prob = log_prob.sum(-1)
    entropy = dist.entropy().sum(-1)
    return action, log_prob, entropy
