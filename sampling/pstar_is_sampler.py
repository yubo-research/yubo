import torch

from sampling.appx_normal import appx_normal
from sampling.hnr import find_perturbation_direction, perturb_normal

"""
TODO
- find best-fit normal distribution, D
- hnr-sample from normal inside bounding box, x_i
- calculate non-appx importance weights (via AppxNormal), w_i
"""


class PStarISSampler:
    def __init__(
        self,
        k_mcmc,
        model,
        *,
        num_X_samples_is=64,
        num_Y_samples=128,
        num_tries=30,
        use_gradients=True,
    ):
        self.model = model
        self.k_mcmc = k_mcmc
        self._eps_interior = torch.tensor(1e-6)
        self._eps_min = 1e-8

        self.appx_normal = appx_normal(
            model=self.model,
            num_X_samples=num_X_samples_is,
            num_Y_samples=num_Y_samples,
            num_tries=num_tries,
            use_gradients=use_gradients,
            seed=None,
        )
        print("AN:", self.appx_normal.mu, self.appx_normal.sigma)

    def __call__(self, num_X_samples):
        with torch.inference_mode():
            return self._sample_pstar(num_X_samples)

    def _sample_pstar(self, num_X_samples):
        X_max = self.appx_normal.mu
        X_max = torch.maximum(self._eps_interior, torch.minimum(1 - self._eps_interior, X_max))
        num_dim = len(X_max.flatten())
        X = torch.tile(X_max, (num_X_samples, 1))

        num_mcmc = num_dim * self.k_mcmc

        eps = 0.01 * float(torch.prod(self.appx_normal.sigma) ** (1 / num_dim))
        num_good = 0
        max_iterations = 10 * num_mcmc
        frac_changed = None
        for i_iter in range(max_iterations):
            i, X_1 = self._hnr_propose(X, eps)
            X[i] = X_1[i]
            frac_changed = (1.0 * i).mean().item()
            # print("FC: eps =", eps, "fc =", frac_changed)
            if abs(frac_changed - 0.5) < 0.1:
                num_good += 1
                # print("GOOD:", num_good)
                if num_good == num_mcmc:
                    print(f"GOODNESS i_iter = {i_iter}")
                    break

            eps = eps * (1 - 1.75 * (0.5 - frac_changed))
            eps = min(1.0, max(self._eps_min, eps))

        else:
            print(
                f"WARNING: Could not determine eps in {max_iterations} iterations. Was at eps = {eps} and num_changed = {num_good}. Last frac_changed = {frac_changed}"
            )

        return X

    def _hnr_propose(self, X, eps):
        X_np = X.detach().numpy()
        u, llambda_minus, llambda_plus = find_perturbation_direction(
            X=X_np,
            num_tries=5,
            eps_bound=min(eps, float(self._eps_interior)) / 100,
        )

        # Make a 1D perturbation
        X_1 = torch.tensor(perturb_normal(X_np, u, eps, llambda_minus, llambda_plus))

        # Metropolis update
        p_0 = self.appx_normal.p_normal(X)
        p_1 = self.appx_normal.p_normal(X_1)
        p_change = torch.min(torch.tensor([1]), p_1 / p_0)
        assert torch.isfinite(p_change.sum()), (
            p_0,
            p_1,
            p_change,
            self.appx_normal.mu,
            self.appx_normal.sigma,
            X,
            X_1,
        )
        u_change = torch.rand(size=p_change.shape)
        b_met_accept = (u_change < p_change).flatten()

        return b_met_accept, X_1
