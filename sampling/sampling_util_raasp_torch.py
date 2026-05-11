import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_samples

from sampling.sampling_util_sobol import _sobol_random_n


def _raasp(x_center, lb, ub, num_candidates, device, dtype):
    num_dim = x_center.shape[-1]
    prob_perturb = min(20.0 / num_dim, 1.0)
    mask = torch.rand(num_candidates, num_dim, device=device) <= prob_perturb

    ind = torch.where(torch.sum(mask, dim=1) == 0)[0]
    if len(ind) > 0:
        mask[ind, torch.randint(0, num_dim, (len(ind),), device=device)] = True

    sobol_samples = draw_sobol_samples(
        bounds=torch.tensor([[0.0] * num_dim, [1.0] * num_dim], dtype=dtype, device=device),
        n=num_candidates,
        q=1,
    ).squeeze(1)

    lb_tensor = torch.tensor(lb, dtype=dtype, device=device)
    ub_tensor = torch.tensor(ub, dtype=dtype, device=device)
    pert = lb_tensor + (ub_tensor - lb_tensor) * sobol_samples

    candidates = x_center.expand(num_candidates, -1)
    candidates = candidates.clone()
    candidates[mask] = pert[mask]

    return candidates


raasp = _raasp


def raasp_turbo_np(x_center, lb, ub, num_candidates, device, dtype):
    num_dim = x_center.shape[-1]
    prob_perturb = min(20.0 / num_dim, 1.0)

    x_center_np = x_center.detach().cpu().numpy()
    lb_np = np.asarray(lb)
    ub_np = np.asarray(ub)

    sobol_samples = _sobol_random_n(
        num_dim,
        num_candidates,
        scramble=True,
        seed=np.random.randint(999999),
    )
    pert = lb_np + (ub_np - lb_np) * sobol_samples

    mask = np.random.rand(num_candidates, num_dim) <= prob_perturb
    ind = np.where(np.sum(mask, axis=1) == 0)[0]
    if len(ind) > 0:
        mask[ind, np.random.randint(0, num_dim, size=len(ind))] = True

    candidates = x_center_np.copy() * np.ones((num_candidates, num_dim))
    candidates[mask] = pert[mask]

    return torch.tensor(candidates, dtype=dtype, device=device)
