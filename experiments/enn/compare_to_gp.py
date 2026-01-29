import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from acq.fit_gp import fit_gp_XY
from problems.env_conf import get_env_conf
from third_party.enn import EpistemicNearestNeighbors, enn_fit
from third_party.enn.enn.enn_params import PosteriorFlags


def compute_gp_ll(model, test_x, test_y_standardized):
    test_x = torch.from_numpy(test_x).to(torch.float64)
    test_y_standardized = torch.from_numpy(test_y_standardized).to(torch.float64)

    with torch.no_grad():
        posterior = model.posterior(test_x)
        output = posterior.distribution
        ll = output.log_prob(test_y_standardized).item()
    return ll / len(test_y_standardized)


def compute_enn_ll(model, params, test_x, test_y_standardized):
    test_x_2d = test_x
    test_y_standardized_2d = test_y_standardized[:, None]

    from third_party.enn.enn.enn_fit import _compute_single_loglik

    flags = PosteriorFlags(observation_noise=False)
    post = model.posterior(test_x_2d, params=params, flags=flags)

    mu_scaled = post.mu
    se_scaled = post.se

    ll = _compute_single_loglik(test_y_standardized_2d, mu_scaled, se_scaled)
    return ll / len(test_y_standardized_2d)


def compute_mean_ll(test_y_standardized):
    from scipy.stats import norm

    # The constant model predicts N(0, 1) in the standardized space
    ll = norm.logpdf(test_y_standardized, loc=0.0, scale=1.0)
    return np.mean(ll)


def sweep_dim_ll_gp_vs_enn(
    func_name: str,
    sigma_noise: float,
    dims: list[int],
    seed: int,
    num_reps: int,
    num_samples_in: int,
    num_samples_out: int,
    k: int,
):
    results = []

    for dim in tqdm(dims, desc=f"Sweeping dims for {func_name}"):
        reps_data = []
        for rep in range(num_reps):
            current_seed = seed + rep
            rng = np.random.default_rng(current_seed + 1)

            # Setup environment with the current seed
            # Use 'f:' prefix for distorted benchmark functions
            env_tag = f"f:{func_name}-{dim}d"
            env_conf = get_env_conf(env_tag, problem_seed=current_seed)
            env = env_conf.make()

            # Generate one noisy data set, then subset it to train & test
            num_samples_total = num_samples_in + num_samples_out
            all_x = rng.uniform(-1, 1, size=(num_samples_total, dim))
            all_r = []
            for x in all_x:
                _, r, _, _ = env.step(x)
                all_r.append(-r)
            all_r = np.array(all_r)
            std_r = np.std(all_r)
            assert np.isfinite(std_r) and std_r > 0
            all_y = all_r + sigma_noise * std_r * rng.standard_normal(size=all_r.shape)
            del all_r

            train_x = all_x[:num_samples_in]
            train_y = all_y[:num_samples_in]

            test_x = all_x[num_samples_in:]
            test_y = all_y[num_samples_in:]

            y_mean_train = np.mean(train_y)
            y_std_train = np.std(train_y)
            train_y_standardized = (train_y - y_mean_train) / y_std_train
            test_y_standardized = (test_y - y_mean_train) / y_std_train
            del train_y, test_y

            # 1. Fit GP
            # Map all_x from [-1, 1] to [0, 1] for BoTorch
            train_x_unit = (train_x + 1.0) / 2.0
            test_x_unit = (test_x + 1.0) / 2.0

            train_x_torch = torch.from_numpy(train_x_unit).to(torch.float64)
            train_y_torch = torch.from_numpy(train_y_standardized[:, None]).to(torch.float64)
            gp_model = fit_gp_XY(train_x_torch, train_y_torch)
            gp_ll = compute_gp_ll(gp_model, test_x_unit, test_y_standardized)

            # 2. Fit ENN
            enn_model = EpistemicNearestNeighbors(train_x, train_y_standardized[:, None])
            enn_params = enn_fit(
                enn_model,
                k=k,
                num_fit_candidates=100,
                num_fit_samples=100,
                rng=rng,
            )
            enn_ll = compute_enn_ll(enn_model, enn_params, test_x, test_y_standardized)

            # 3. Baseline Mean Model
            mean_ll = compute_mean_ll(test_y_standardized)

            reps_data.append((gp_ll, enn_ll, mean_ll))

        gp_lls, enn_lls, mean_lls = zip(*reps_data)
        results.append(
            {
                "num_dim": dim,
                "ll_gp_mean": np.mean(gp_lls),
                "ll_gp_se": np.std(gp_lls) / np.sqrt(num_reps),
                "ll_enn_mean": np.mean(enn_lls),
                "ll_enn_se": np.std(enn_lls) / np.sqrt(num_reps),
                "ll_mean_mean": np.mean(mean_lls),
                "ll_mean_se": np.std(mean_lls) / np.sqrt(num_reps),
                "func": func_name,
            }
        )

    return pd.DataFrame(results)
