import torch
from torch import Tensor

from model.enn_gp_t import EpistemicNearestNeighborsGP
from sampling.diversity_subsample import diversity_subsample


def subsample_indices_with_extremes(n: int, P: int, y: Tensor, device: torch.device) -> Tensor:
    assert n > 0
    assert P > 0
    assert y.ndim == 1
    assert y.shape[0] == n

    P_actual = min(P, n)
    if P_actual == n:
        return torch.arange(n, device=device)

    y_min_idx = y.argmin()
    y_max_idx = y.argmax()
    required = torch.unique(torch.stack([y_min_idx, y_max_idx]))

    if P_actual <= len(required):
        return required[:P_actual]

    remaining_mask = torch.ones(n, dtype=torch.bool, device=device)
    remaining_mask[required] = False
    remaining = torch.where(remaining_mask)[0]
    num_needed = P_actual - len(required)

    if num_needed > 0 and len(remaining) > 0:
        perm = torch.randperm(len(remaining), device=device)[:num_needed]
        sampled = remaining[perm]
        return torch.cat([required, sampled])

    return required


def subsample_loglik(
    model: EpistemicNearestNeighborsGP,
    x: Tensor,
    y: Tensor,
    y_var: Tensor | None = None,
    P: int = 10,
    subsample_type: str = "plain",
) -> Tensor:
    assert x.ndim == 2
    assert y.ndim == 1
    assert y.shape[0] == x.shape[0]
    assert P > 0
    assert subsample_type in ("plain", "plainex", "diverse"), f"subsample_type must be 'plain', 'plainex', or 'diverse', got {subsample_type}"
    if y_var is None:
        y_var = torch.zeros_like(y)
    assert y_var.shape == y.shape
    n = x.shape[0]
    if n == 0:
        return torch.as_tensor(0.0, dtype=x.dtype, device=x.device)
    P_actual = min(P, n)
    if P_actual == n:
        indices = torch.arange(n, device=x.device)
    elif subsample_type == "plain":
        indices = torch.randperm(n, device=x.device)[:P_actual]
    elif subsample_type == "plainex":
        indices = subsample_indices_with_extremes(n, P, y, x.device)
    elif subsample_type == "diverse":
        x_np = x.detach().cpu().numpy()
        indices_np = diversity_subsample(x_np, P_actual)
        indices = torch.from_numpy(indices_np).to(device=x.device, dtype=torch.long)
    x_selected = x[indices]
    y_selected = y[indices]

    if not torch.isfinite(y_selected).all():
        return torch.as_tensor(0.0, dtype=x.dtype, device=x.device)

    exclude_nearest = len(model._enn) > 1
    posterior = model.posterior(x_selected, exclude_nearest=exclude_nearest, observation_noise=True)
    mvn = posterior.distribution
    mu = mvn.mean.squeeze(-1)
    se = mvn.variance.sqrt()

    if not torch.isfinite(mu).all() or not torch.isfinite(se).all() or (se <= 0).any():
        return torch.as_tensor(0.0, dtype=x.dtype, device=x.device)

    normal = torch.distributions.Normal(mu, se)
    loglik = normal.log_prob(y_selected).sum()
    # print("MSE:", ((mu - y_selected) ** 2).mean().item(), "loglik =", loglik.item())

    if not torch.isfinite(loglik):
        loglik = torch.as_tensor(0.0, dtype=loglik.dtype, device=loglik.device)
    # print("X:", loglik.item(), k_int, var_scale_float)
    return loglik
