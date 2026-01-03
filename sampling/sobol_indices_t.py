import torch


def calculate_sobol_indices_t(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    n, d = x.shape
    assert d > 0
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.view(-1)
    assert y.ndim == 1
    assert y.shape[0] == n
    if n < 9:
        return torch.ones(d, dtype=x.dtype, device=x.device)
    mu = y.mean()
    vy = y.var(unbiased=False)
    if not torch.isfinite(vy) or vy <= 0:
        return torch.ones(d, dtype=x.dtype, device=x.device)
    B = 10 if n >= 30 else 3
    order = torch.argsort(x, dim=0)
    row_idx = torch.arange(n, device=x.device).unsqueeze(1).expand(n, d)
    ranks = torch.empty_like(order)
    ranks.scatter_(0, order, row_idx)
    idx = (ranks * B) // n
    oh = torch.nn.functional.one_hot(idx.to(torch.long), num_classes=B).to(dtype=x.dtype)
    counts = oh.sum(dim=0)
    sums = (oh * y.view(n, 1, 1)).sum(dim=0)
    mu_b = torch.zeros_like(sums)
    mask = counts > 0
    mu_b[mask] = sums[mask] / counts[mask]
    p_b = counts / float(n)
    diff = mu_b - mu
    S = (p_b * (diff * diff)).sum(dim=1) / vy
    var_x = x.var(dim=0, unbiased=False)
    S = S.to(dtype=x.dtype)
    S = torch.where(var_x <= torch.as_tensor(1e-12, dtype=x.dtype, device=x.device), torch.zeros_like(S), S)
    return S
