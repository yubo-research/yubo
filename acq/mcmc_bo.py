from dataclasses import dataclass

import numpy as np
import torch
from torch.quasirandom import SobolEngine


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = np.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

    def update_state(self, Y_next):
        if max(Y_next) > self.best_value + 1e-3 * np.fabs(self.best_value):
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter == self.success_tolerance:  # Expand trust region
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:  # Shrink trust region
            self.length /= 2.0
            self.failure_counter = 0

        self.best_value = max(self.best_value, max(Y_next).item())
        if self.length < self.length_min:
            self.restart_triggered = True
        return self


def cdf(value, mean=0, std=1):
    return 0.5 * (1 + torch.erf((value - mean) / (std * np.sqrt(2))))


def get_point_in_tr(x, lb, ub):
    assert x.ndim == 1
    x = x.detach().cpu().numpy()
    return np.all(x > lb) and np.all(x < ub)


def mcmc_one_transit(original_x, noise_size, model, dtype, device, lb=0, ub=1):
    x_set = torch.zeros((original_x.shape[0], 2, original_x.shape[1])).to(
        dtype=dtype, device=device
    )
    x_set[:, 0, :] = original_x
    x_set[:, 1, :] = original_x + noise_size * torch.randn(original_x.shape).to(
        dtype=dtype, device=device
    )
    # reject points out of trust region
    for i, x in enumerate(x_set):
        if not get_point_in_tr(x_set[i, 1, :], lb, ub):
            x_set[i, 1, :] = x_set[i, 0, :]
    x_set_dis = model(x_set)
    a = torch.tensor([1.0, -1.0]).to(dtype=dtype, device=device)
    new_cov = torch.matmul(x_set_dis.lazy_covariance_matrix.matmul(a), a.t())
    new_cov[new_cov < 0] = 0
    mean = x_set_dis.mean[:, 0] - x_set_dis.mean[:, 1]

    temp = (torch.zeros(mean.shape).cuda() - mean) / (torch.sqrt(new_cov + 1e-6))
    m = torch.distributions.normal.Normal(
        torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda()
    )
    p = m.cdf(temp)  # normally distributed with loc=0 and scale=1
    p_ratio = (p) / (1 - p + 1e-7)
    alpha = torch.clamp(p_ratio, 0, 1)
    sample = torch.rand(alpha.shape).to(dtype=dtype, device=device)
    idx = torch.zeros(alpha.shape).to(dtype=dtype, device=device)
    idx[sample < alpha] = 1
    new_x = x_set[idx == 1, 1]
    old_x = x_set[idx == 0, 0]
    new_pop = torch.cat((old_x, new_x), dim=0)

    return new_pop


def langevin_update(
    x_cur, langevin_epsilon, model, dtype, device, lb=0, ub=1, h=5e-5, n_splits=4
):
    beta = 2
    n, d = x_cur.shape[0], x_cur.shape[1]
    if n > n_splits:
        batch_size = n // n_splits
    else:
        batch_size = n
    gradients_all = torch.zeros_like(x_cur)
    for idx in range(0, n, batch_size):
        aug_x = (
            x_cur[idx : idx + batch_size, :]
            .unsqueeze(1)
            .unsqueeze(1)
            .repeat(1, d, 2, 1)
        )
        for dim in range(d):
            aug_x[:, dim, 1, dim] += h

        posterior = model.posterior(aug_x)
        f_mean, f_covar = posterior.mvn.mean, posterior.mvn.covariance_matrix
        Sigma = f_covar.detach()
        Sigma[:, :, 0, 0] += 1e-4
        Sigma[:, :, 1, 1] += 1e-4
        Sigma_nd = (
            Sigma[:, :, 0, 0]
            + Sigma[:, :, 1, 1]
            - Sigma[:, :, 1, 0]
            - Sigma[:, :, 0, 1]
        )
        mu_nd = (
            f_mean[:, :, 0]
            - f_mean[:, :, 1]
            + beta * (Sigma[:, :, 0, 0] - Sigma[:, :, 1, 1])
        )
        x_grad = mu_nd / torch.sqrt(4 * Sigma_nd)
        x_grad = cdf(-1 * x_grad)
        try:
            x_grad = ((x_grad / (1 - x_grad)) - 1) / h  # (n, d)b``
        except ZeroDivisionError:
            print(f"ZeroDivisionError: {x_grad}")
        gradients_all[idx : idx + batch_size, :] = x_grad
    noise = torch.randn_like(x_cur, device=device) * torch.sqrt(
        torch.Tensor([2 * langevin_epsilon]).to(device=device)
    )
    x_cur = x_cur + langevin_epsilon * gradients_all + noise  # (n, d)
    if torch.isnan(gradients_all).any():
        raise AssertionError("Gradient nan")
    x_next = torch.clamp(x_cur, torch.tensor(lb).cuda(), torch.tensor(ub).cuda())
    return x_next


def generate_batch_multiple_tr(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    dtype,
    device,
    num_candidates=None,  # Number of candidates for Thompson sampling
    acqf="ts",  # "ei" or "ts"
    mcmc_type=None,
    # use_langevin=False
):
    assert acqf in ("ts", "ei")
    tr_num = len(state)

    for tr_idx in range(tr_num):
        assert (
            X[tr_idx].min() >= 0.0
            and X[tr_idx].max() <= 1.0
            and torch.all(torch.isfinite(Y[tr_idx]))
        )
    if num_candidates is None:
        num_candidates = min(5000, max(2000, 200 * X.shape[-1]))
    dim = X[0].shape[1]
    # Scale the TR to be proportional to the lengthscales
    X_cand = torch.zeros(tr_num, num_candidates, dim).to(device=device, dtype=dtype)
    Y_cand = torch.zeros(tr_num, num_candidates, batch_size).to(
        device=device, dtype=dtype
    )
    tr_lb = torch.zeros(tr_num, dim).to(device=device, dtype=dtype)
    tr_ub = torch.zeros(tr_num, dim).to(device=device, dtype=dtype)
    for tr_idx in range(tr_num):
        x_center = X[tr_idx][Y[tr_idx].argmax(), :].clone()
        try:
            weights = (
                model[tr_idx].covar_module.base_kernel.lengthscale.squeeze().detach()
            )
            weights = weights / weights.mean()
            weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
            tr_lb[tr_idx] = torch.clamp(
                x_center - weights * state[tr_idx].length / 2.0, 0.0, 1.0
            )
            tr_ub[tr_idx] = torch.clamp(
                x_center + weights * state[tr_idx].length / 2.0, 0.0, 1.0
            )
        except Exception:  # Linear kernel
            weights = 1
            tr_lb[tr_idx] = torch.clamp(x_center - state[tr_idx].length / 2.0, 0.0, 1.0)
            tr_ub[tr_idx] = torch.clamp(x_center + state[tr_idx].length / 2.0, 0.0, 1.0)

        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(num_candidates).to(dtype=dtype, device=device)
        pert = tr_lb[tr_idx] + (tr_ub[tr_idx] - tr_lb[tr_idx]) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        # prob_perturb = 1
        mask = (
            torch.rand(num_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        if dim == 1:
            rr = torch.zeros(size=(len(ind),), device=device, dtype=torch.int64)
        else:
            rr = torch.randint(0, dim - 1, size=(len(ind),), device=device)
        mask[ind, rr] = 1

        # Create candidate points from the perturbations and the mask
        X_cand[tr_idx] = x_center.expand(num_candidates, dim).clone()
        X_cand[tr_idx][mask] = pert[mask]

        # Sample on the candidate points
        posterior = model[tr_idx].posterior(X_cand[tr_idx])
        samples = posterior.rsample(sample_shape=torch.Size([batch_size]))
        samples = samples.reshape([batch_size, num_candidates])
        Y_cand[tr_idx] = samples.permute(1, 0)
        # recover from normalized value
        Y_cand[tr_idx] = Y[tr_idx].mean() + Y_cand[tr_idx] * Y[tr_idx].std()

    # Compare across trust region
    y_cand = Y_cand.detach().cpu().numpy()
    X_next = torch.zeros(batch_size, dim).to(device=device, dtype=dtype)
    tr_idx_next = np.zeros(batch_size)
    for k in range(batch_size):
        i, j = np.unravel_index(np.argmax(y_cand[:, :, k]), (tr_num, num_candidates))
        X_next[k] = X_cand[i, j]
        tr_idx_next[k] = i
        assert np.isfinite(
            y_cand[i, j, k]
        )  # Just to make sure we never select nan or inf
        # Make sure we never pick this point again
        y_cand[i, j, :] = -np.inf

    if mcmc_type == "MH":
        for tr_idx in range(tr_num):
            noise_size = state[tr_idx].length * weights / 2
            idx_in_tr = np.argwhere(tr_idx_next == tr_idx).reshape(-1)
            if idx_in_tr.shape[0] == 0:
                continue
            with torch.no_grad():  # We don't need gradients when using TS
                mcmc_round = max(200, dim)
                for i in range(mcmc_round):
                    new_pop = mcmc_one_transit(
                        X_next[idx_in_tr],
                        0.001 * noise_size,
                        model[tr_idx],
                        tr_lb[tr_idx].detach().cpu().numpy(),
                        tr_ub[tr_idx].detach().cpu().numpy(),
                    )
                    X_next[idx_in_tr] = new_pop
    elif mcmc_type == "Langevin":
        for tr_idx in range(tr_num):
            noise_size = state[tr_idx].length
            idx_in_tr = np.argwhere(tr_idx_next == tr_idx).reshape(-1)
            if idx_in_tr.shape[0] == 0:
                continue
            with torch.no_grad():  # We don't need gradients when using TS
                round = max(200, dim)
                for i in range(round):
                    new_pop = langevin_update(
                        X_next[idx_in_tr],
                        2e-3 * noise_size,
                        model[tr_idx],
                        tr_lb[tr_idx].detach().cpu().numpy(),
                        tr_ub[tr_idx].detach().cpu().numpy(),
                    )
                    X_next[idx_in_tr] = new_pop
    else:
        assert mcmc_type is None, mcmc_type

    return X_next, tr_idx_next
