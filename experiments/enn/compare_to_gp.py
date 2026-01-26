import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from acq.fit_gp import fit_gp_XY
from problems.env_conf import get_env_conf
from third_party.enn import EpistemicNearestNeighbors, enn_fit
from third_party.enn.enn.enn_params import PosteriorFlags


def compute_gp_ll(model, test_x, test_y):
    test_x = torch.from_numpy(test_x).to(torch.float64)
    test_y = torch.from_numpy(test_y).to(torch.float64)

    # Standardize test_y using the same logic as fit_gp_XY
    # fit_gp_XY calls standardize_torch(Y) which uses botorch.utils.standardize
    from acq.fit_gp import standardize_torch

    test_y_standardized = standardize_torch(test_y[:, None]).squeeze(-1)

    with torch.no_grad():
        posterior = model.posterior(test_x)
        output = posterior.distribution
        ll = output.log_prob(test_y_standardized).item()
    return ll / len(test_y)


def compute_enn_ll(model, params, test_x, test_y):
    test_x_2d = test_x
    test_y_2d = test_y[:, None]

    from third_party.enn.enn.enn_fit import _compute_single_loglik

    flags = PosteriorFlags(observation_noise=True)
    post = model.posterior(test_x_2d, params=params, flags=flags)

    y_std = np.std(model.train_y, axis=0, keepdims=True)
    y_std = np.where(np.isfinite(y_std) & (y_std > 0.0), y_std, 1.0)

    y_scaled = test_y_2d / y_std
    mu_scaled = post.mu / y_std
    se_scaled = post.se / y_std

    ll = _compute_single_loglik(y_scaled, mu_scaled, se_scaled)
    return ll / len(test_y)


def sweep_dim_ll_gp_vs_enn(
    func_name: str,
    sigma_noise: float,
    dims: list[int],
    seed: int,
    num_reps: int,
    num_samples_in: int,
    num_samples_out: int,
):
    results = []

    for dim in tqdm(dims, desc=f"Sweeping dims for {func_name}"):
        reps_data = []
        for rep in range(num_reps):
            current_seed = seed + rep
            rng = np.random.default_rng(current_seed)

            # Setup environment with the current seed
            # Use 'f:' prefix for distorted benchmark functions
            env_tag = f"f:{func_name}-{dim}d"
            env_conf = get_env_conf(env_tag, problem_seed=current_seed)
            env = env_conf.make()

            # Generate training data
            train_x = rng.uniform(-1, 1, size=(num_samples_in, dim))
            train_r = []
            for x in train_x:
                _, r, _, _ = env.step(x)
                train_r.append(-r)
            train_r = np.array(train_r)
            std_r = np.std(train_r)
            assert np.isfinite(std_r) and std_r > 0
            train_y = train_r + sigma_noise * std_r * rng.standard_normal(
                size=train_r.shape
            )

            # Generate test data
            test_x = rng.uniform(-1, 1, size=(num_samples_out, dim))
            test_y = []
            for x in test_x:
                _, r, _, _ = env.step(x)
                test_y.append(-r)
            test_y = np.array(test_y)

            # 1. Fit GP
            train_x_torch = torch.from_numpy(train_x).to(torch.float64)
            train_y_torch = torch.from_numpy(train_y[:, None]).to(torch.float64)
            gp_model = fit_gp_XY(train_x_torch, train_y_torch)
            gp_ll = compute_gp_ll(gp_model, test_x, test_y)

            # 2. Fit ENN
            train_y_2d = train_y[:, None]
            enn_model = EpistemicNearestNeighbors(train_x, train_y_2d)
            enn_params = enn_fit(
                enn_model,
                k=10,
                num_fit_candidates=100,
                num_fit_samples=100,
                rng=rng,
            )
            enn_ll = compute_enn_ll(enn_model, enn_params, test_x, test_y)

            reps_data.append((gp_ll, enn_ll))

        gp_lls, enn_lls = zip(*reps_data)
        results.append(
            {
                "num_dim": dim,
                "ll_gp_mean": np.mean(gp_lls),
                "ll_gp_se": np.std(gp_lls) / np.sqrt(num_reps),
                "ll_enn_mean": np.mean(enn_lls),
                "ll_enn_se": np.std(enn_lls) / np.sqrt(num_reps),
                "func": func_name,
            }
        )

    return pd.DataFrame(results)
