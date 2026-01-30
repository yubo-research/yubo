import numpy as np
import torch
from botorch.sampling.qmc import MultivariateNormalQMCEngine
from botorch.utils.sampling import draw_sobol_samples
from scipy.stats import multivariate_normal, qmc
from torch.quasirandom import SobolEngine


def greedy_maximin(x, num_subsamples):
    # Credit to ChatGPT

    num_samples = x.shape[0]
    assert num_subsamples < num_samples, (num_samples, num_subsamples)

    selected_indices = []

    first_idx = np.random.choice(num_samples)
    selected_indices.append(first_idx)

    diff = x - x[first_idx]
    min_dists = np.linalg.norm(diff, axis=1)
    min_dists[first_idx] = 0

    for _ in range(1, num_subsamples):
        next_idx = np.argmax(min_dists)
        selected_indices.append(next_idx)

        dists = np.linalg.norm(x - x[next_idx], axis=1)
        min_dists = np.minimum(min_dists, dists)

    return selected_indices


def top_k(x, k):
    assert len(x.shape) == 1, x.shape

    if k >= len(x):
        return np.argsort(x)[::-1]

    return np.argpartition(x, -k)[-k:]


def intersect_with_box(x_inside, x_outside):
    t_min, t_max = 0, 1  # Initialize the range of valid t values
    for i in range(len(x_inside)):
        t0 = (0 - x_inside[i]) / (
            x_outside[i] - x_inside[i]
        )  # Intersection with the lower boundary
        t1 = (1 - x_inside[i]) / (
            x_outside[i] - x_inside[i]
        )  # Intersection with the upper boundary
        t0, t1 = min(t0, t1), max(t0, t1)  # Order t0 and t1
        t_min = max(t_min, t0)  # Update the lower bound of t
        t_max = min(t_max, t1)  # Update the upper bound of t

    print("T:", t_min, t_max)
    if t_min > t_max:
        return None  # No valid intersection
    print("D:", x_outside - x_inside, t_min * (x_outside - x_inside))
    return x_inside + t_min * (x_outside - x_inside)


def var_of_var(w: torch.Tensor, X: torch.Tensor):
    assert torch.abs(w.sum() - 1) < 1e-6, w.sum()
    mu = (w * X**2).sum(dim=0)
    dev = X - mu
    return var_of_var_dev(w, dev)


def var_of_var_dev(w: torch.Tensor, dev: torch.Tensor):
    assert torch.abs(w.sum() - 1) < 1e-6, w.sum()
    mu_2 = (w * dev**2).sum(dim=0)
    mu_4 = (w * dev**4).sum(dim=0)
    n = len(dev)
    return (mu_4 - mu_2**2) / n


def qmc_normal_sample(mu, cov, num_samples=1):
    qmcn = MultivariateNormalQMCEngine(
        torch.tensor(mu),
        torch.tensor(np.diag(cov)),
    )
    return qmcn.draw(num_samples)


def draw_bounded_normal_samples(mu, cov, num_samples, qmc=False):
    rv_norm = multivariate_normal(
        mean=mu,
        cov=cov,
    )

    num_dim = len(mu)

    if qmc:
        x_n = qmc_normal_sample(mu, cov, num_samples).numpy()
        sobol_engine = SobolEngine(num_dim, scramble=True)
        x_u = sobol_engine.draw(num_samples).numpy()
    else:
        x_n = rv_norm.rvs(size=(2 * num_samples,))
        if num_dim == 1:
            x_n = x_n[:, None]
        x_u = np.random.uniform(size=(2 * num_samples, len(mu)))

    p_n = rv_norm.pdf(x_n)
    p_n = p_n / p_n.sum() / 2
    p_u = np.ones(shape=(len(x_u),))
    p_u = p_u / p_u.sum() / 2

    i = np.where((x_n.min(axis=1) >= 0) & (x_n.max(axis=1) <= 1))[0]
    x_n = x_n[i, :]
    p_n = p_n[i]

    x = np.concatenate((x_n, x_u), axis=0)
    p = np.concatenate((p_n, p_u), axis=0)
    p = rv_norm.pdf(x) / p
    p = p / p.sum()

    i = np.random.choice(np.arange(len(x)), p=p, size=(num_samples,))
    x = x[i, :]
    p = p[i]
    p = p / p.sum()

    return x, p


def _xxx_draw_varied_bounded_normal_samples(mus_covs):
    samples = []
    num_dim = len(mus_covs[0][0])
    for mu, cov in mus_covs:
        assert len(mu) == num_dim, (len(mu), num_dim)

        rv_norm = multivariate_normal(
            mean=mu,
            cov=cov,
        )

        x = rv_norm.rvs()
        if num_dim == 1:
            x = np.array([x])
        x = x[:, None]
        if x.min() < 0 or x.max() > 1:
            x = np.random.uniform(size=(1, len(mu)))
        p = rv_norm.pdf(x.flatten())
        samples.append((x, p))

    samples = np.array(samples)
    x = np.stack(samples[:, 0]).squeeze(-1)
    p = samples[:, 1]

    return x, p


def raasp_np_choice(
    x_center,
    lb,
    ub,
    num_candidates,
):
    num_dim = x_center.shape[-1]
    k = min(20, num_dim)

    sobol_engine = qmc.Sobol(num_dim, scramble=True)
    sobol_samples = sobol_engine.random(num_candidates)

    lb_array = np.asarray(lb)
    ub_array = np.asarray(ub)
    pert = lb_array + (ub_array - lb_array) * sobol_samples

    if k == num_dim:
        return pert
    else:
        candidates = np.tile(x_center, (num_candidates, 1))
        rand = np.random.rand(num_candidates, num_dim)
        dims = np.argpartition(rand, k, axis=1)[:, :k]
        row_idx = np.repeat(np.arange(num_candidates), k)
        col_idx = dims.ravel()
        candidates[row_idx, col_idx] = pert[row_idx, col_idx]
        return candidates


def raasp_np_p(x_center, lb, ub, num_candidates, i_dim_allowed=None, stagger=False):
    num_dim = x_center.shape[-1]
    return raasp_np(
        x_center,
        lb,
        ub,
        num_candidates,
        num_pert=min(20, int(num_dim * 0.20 + 0.5)),
        i_dim_allowed=i_dim_allowed,
        stagger=stagger,
    )


def raasp_np(
    x_center, lb, ub, num_candidates, num_pert=20, *, i_dim_allowed=None, stagger=False
):
    num_dim = x_center.shape[-1]

    if i_dim_allowed is not None:
        prob_perturb = np.zeros(num_dim)
        prob_perturb[i_dim_allowed] = min(num_pert / len(i_dim_allowed), 1.0)
    else:
        prob_perturb = min(num_pert / num_dim, 1.0)

    mask = np.random.rand(num_candidates, num_dim) <= prob_perturb

    ind = np.where(np.sum(mask, axis=1) == 0)[0]
    if len(ind) > 0:
        mask[ind, np.random.randint(0, num_dim, size=len(ind))] = True

    return sobol_perturb_np(x_center, lb, ub, num_candidates, mask, stagger=stagger)


def raasp_np_1d(
    x_centers: np.ndarray, lb: np.ndarray, ub: np.ndarray, num_candidates: int
) -> np.ndarray:
    # TODO: Return the corresponding centers, too.
    assert len(x_centers.shape) == 2, (
        f"x_centers must be 2D, got shape {x_centers.shape}"
    )
    assert len(lb.shape) == 2 and lb.shape[0] == 1, (
        f"lb must have shape (1, num_dim), got shape {lb.shape}"
    )
    assert len(ub.shape) == 2 and ub.shape[0] == 1, (
        f"ub must have shape (1, num_dim), got shape {ub.shape}"
    )
    assert x_centers.shape[1] == lb.shape[1], (
        f"x_centers and lb must have same num_dim, got {x_centers.shape[1]} and {lb.shape[1]}"
    )
    assert lb.shape[1] == ub.shape[1], (
        f"lb and ub must have same num_dim, got {lb.shape[1]} and {ub.shape[1]}"
    )
    assert num_candidates > 0, f"num_candidates must be positive, got {num_candidates}"

    num_centers, num_dim = x_centers.shape

    center_indices = np.arange(num_candidates) % num_centers
    x_candidates = x_centers[center_indices].copy()

    dim_indices = np.random.randint(0, num_dim, size=num_candidates)
    row_indices = np.arange(num_candidates)

    sobol_engine = qmc.Sobol(num_dim, scramble=True, seed=np.random.randint(999999))
    sobol_samples = sobol_engine.random(num_candidates)
    perturbations = lb + (ub - lb) * sobol_samples

    x_candidates[row_indices, dim_indices] = perturbations[row_indices, dim_indices]

    return x_candidates


def truncated_normal_np(
    mu: np.ndarray, sigma: np.ndarray, lb: np.ndarray, ub: np.ndarray, num_candidates
):
    assert mu.shape == sigma.shape == lb.shape == ub.shape, (
        f"All inputs must have same shape, got mu: {mu.shape}, sigma: {sigma.shape}, lb: {lb.shape}, ub: {ub.shape}"
    )
    assert len(mu.shape) == 2 and mu.shape[0] == 1, (
        f"All inputs must have shape (1, num_dim), got {mu.shape}"
    )
    num_dim = mu.shape[1]
    assert np.all(sigma >= 0), "All sigma values must be non-negative"
    assert np.all(lb < ub), "All lb values must be less than ub values"

    from scipy.stats import truncnorm

    a = (lb - mu) / sigma
    b = (ub - mu) / sigma

    rv = truncnorm(a.flatten(), b.flatten(), loc=mu.flatten(), scale=sigma.flatten())
    samples = rv.rvs(size=(num_candidates, num_dim))

    return samples


def sobol_perturb_np(x_center, lb, ub, num_candidates, mask, stagger=False):
    num_dim = x_center.shape[-1]
    sobol_engine = qmc.Sobol(num_dim, scramble=True, seed=np.random.randint(999999))
    sobol_samples = sobol_engine.random(num_candidates)
    lb_array = np.asarray(lb)
    ub_array = np.asarray(ub)
    pert = lb_array + (ub_array - lb_array) * sobol_samples

    candidates = np.tile(x_center, (num_candidates, 1))

    if np.any(mask):
        if stagger:
            assert False
            l_s_min = np.log(1e-4)
            l_s_max = np.log(1.0)
            alpha = np.random.uniform(0, 1, size=(num_candidates, num_dim))
            alpha = np.exp(l_s_min + (l_s_max - l_s_min) * alpha)
        else:
            alpha = np.ones((num_candidates, num_dim))

        candidates[mask] = candidates[mask] + alpha[mask] * (
            pert[mask] - candidates[mask]
        )

    return candidates


def raasp(x_center, lb, ub, num_candidates, device, dtype):
    num_dim = x_center.shape[-1]
    prob_perturb = min(20.0 / num_dim, 1.0)
    mask = torch.rand(num_candidates, num_dim, device=device) <= prob_perturb

    ind = torch.where(torch.sum(mask, dim=1) == 0)[0]
    if len(ind) > 0:
        mask[ind, torch.randint(0, num_dim, (len(ind),), device=device)] = True

    sobol_samples = draw_sobol_samples(
        bounds=torch.tensor(
            [[0.0] * num_dim, [1.0] * num_dim], dtype=dtype, device=device
        ),
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


def raasp_turbo_np(x_center, lb, ub, num_candidates, device, dtype):
    num_dim = x_center.shape[-1]
    prob_perturb = min(20.0 / num_dim, 1.0)

    x_center_np = x_center.detach().cpu().numpy()
    lb_np = np.asarray(lb)
    ub_np = np.asarray(ub)

    sobol = qmc.Sobol(num_dim, scramble=True, seed=np.random.randint(999999))
    sobol_samples = sobol.random(num_candidates)
    pert = lb_np + (ub_np - lb_np) * sobol_samples

    mask = np.random.rand(num_candidates, num_dim) <= prob_perturb
    ind = np.where(np.sum(mask, axis=1) == 0)[0]
    if len(ind) > 0:
        mask[ind, np.random.randint(0, num_dim, size=len(ind))] = True

    candidates = x_center_np.copy() * np.ones((num_candidates, num_dim))
    candidates[mask] = pert[mask]

    return torch.tensor(candidates, dtype=dtype, device=device)


def gumbel(n):
    if n <= 0:
        return 0.0
    if n == 1:
        return 0.0

    log_n = np.log(n)
    log_log_n = np.log(log_n)
    return np.sqrt(2 * log_n) - (log_log_n + np.log(4 * np.pi)) / (
        2 * np.sqrt(2 * log_n)
    )
