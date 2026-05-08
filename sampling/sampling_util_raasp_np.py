import numpy as np

from sampling.sampling_util_sobol import _sobol_random_n


def raasp_np_choice(
    x_center,
    lb,
    ub,
    num_candidates,
):
    num_dim = x_center.shape[-1]
    k = min(20, num_dim)

    sobol_samples = _sobol_random_n(num_dim, num_candidates, scramble=True)

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


def raasp_np(x_center, lb, ub, num_candidates, num_pert=20, *, i_dim_allowed=None, stagger=False):
    num_dim = x_center.shape[-1]

    if i_dim_allowed is not None:
        if len(i_dim_allowed) == 0:
            raise ValueError("i_dim_allowed must be non-empty when provided")
        prob_perturb = np.zeros(num_dim)
        prob_perturb[i_dim_allowed] = min(num_pert / len(i_dim_allowed), 1.0)
    else:
        prob_perturb = min(num_pert / num_dim, 1.0)

    mask = np.random.rand(num_candidates, num_dim) <= prob_perturb

    ind = np.where(np.sum(mask, axis=1) == 0)[0]
    if len(ind) > 0:
        mask[ind, np.random.randint(0, num_dim, size=len(ind))] = True

    return sobol_perturb_np(x_center, lb, ub, num_candidates, mask, stagger=stagger)


def raasp_np_1d(x_centers: np.ndarray, lb: np.ndarray, ub: np.ndarray, num_candidates: int) -> np.ndarray:
    # TODO: Return the corresponding centers, too.
    assert len(x_centers.shape) == 2, f"x_centers must be 2D, got shape {x_centers.shape}"
    assert len(lb.shape) == 2 and lb.shape[0] == 1, f"lb must have shape (1, num_dim), got shape {lb.shape}"
    assert len(ub.shape) == 2 and ub.shape[0] == 1, f"ub must have shape (1, num_dim), got shape {ub.shape}"
    assert x_centers.shape[1] == lb.shape[1], f"x_centers and lb must have same num_dim, got {x_centers.shape[1]} and {lb.shape[1]}"
    assert lb.shape[1] == ub.shape[1], f"lb and ub must have same num_dim, got {lb.shape[1]} and {ub.shape[1]}"
    assert num_candidates > 0, f"num_candidates must be positive, got {num_candidates}"

    num_centers, num_dim = x_centers.shape

    center_indices = np.arange(num_candidates) % num_centers
    x_candidates = x_centers[center_indices].copy()

    dim_indices = np.random.randint(0, num_dim, size=num_candidates)
    row_indices = np.arange(num_candidates)

    sobol_samples = _sobol_random_n(
        num_dim,
        num_candidates,
        scramble=True,
        seed=np.random.randint(999999),
    )
    perturbations = lb + (ub - lb) * sobol_samples

    x_candidates[row_indices, dim_indices] = perturbations[row_indices, dim_indices]

    return x_candidates


def truncated_normal_np(mu: np.ndarray, sigma: np.ndarray, lb: np.ndarray, ub: np.ndarray, num_candidates):
    assert mu.shape == sigma.shape == lb.shape == ub.shape, (
        f"All inputs must have same shape, got mu: {mu.shape}, sigma: {sigma.shape}, lb: {lb.shape}, ub: {ub.shape}"
    )
    assert len(mu.shape) == 2 and mu.shape[0] == 1, f"All inputs must have shape (1, num_dim), got {mu.shape}"
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
    sobol_samples = _sobol_random_n(
        num_dim,
        num_candidates,
        scramble=True,
        seed=np.random.randint(999999),
    )
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

        candidates[mask] = candidates[mask] + alpha[mask] * (pert[mask] - candidates[mask])

    return candidates


def gumbel(n):
    if n <= 0:
        return 0.0
    if n == 1:
        return 0.0

    log_n = np.log(n)
    log_log_n = np.log(log_n)
    return np.sqrt(2 * log_n) - (log_log_n + np.log(4 * np.pi)) / (2 * np.sqrt(2 * log_n))
