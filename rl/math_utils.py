import torch


def atanh(x: torch.Tensor, eps: float = 1e-06) -> torch.Tensor:
    """Inverse hyperbolic tangent with clamping for numerical stability."""
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))
