import time

import numpy as np
import torch

# from sampling.appx_normal import appx_normal
from sampling.appx_trunc_normal import appx_trunc_normal
from sampling.hnr import find_perturbation_direction, perturb_normal
from sampling.mv_truncated_normal import MVTruncatedNormal
from sampling.parallel_mcmc_convergence import ParallelMCMCConvergence


class PStarISSampler:
    def __init__(
        self,
        k_mcmc,
        model,
        *,
        num_X_samples_appx_normal=64,
        num_Y_samples=1024,
        num_tries=3,
        use_gradients=True,
        theta=np.inf,
    ):
        self.model = model
        self.k_mcmc = k_mcmc
        self._eps_interior = torch.tensor(1e-6)
        self._eps_min = 1e-8

        # Approximate p*(x) by a normal distribution.
        t_0 = time.time()
        self.appx_normal = appx_trunc_normal(
            model=self.model,
            num_X_samples=num_X_samples_appx_normal,
            num_Y_samples=num_Y_samples,
            num_tries=num_tries,
            use_gradients=use_gradients,
            seed=None,
            theta=theta,
        )
        t_f = time.time()
        print(f"APPX_NORMAL: dt = {t_f-t_0:.2}s")
        # print("AN:", self.appx_normal.mu, self.appx_normal.sigma)

    def __call__(self, num_X_samples):
        # t_0 = time.time()
        with torch.inference_mode():
            X_samples = torch.as_tensor(self._sample_pstar(num_X_samples))
        # t_f = time.time()
        # print(f"SAMPLE: dt = {t_f-t_0:.2}s")
        weights = self.appx_normal.calc_importance_weights(X_samples)
        return weights, X_samples

    def _sample_pstar(self, num_X_samples):
        # Sample from the approximate p*(x) within the bounding box.

        return MVTruncatedNormal(
            loc=self.appx_normal.mu,
            scale=self.appx_normal.sigma,
        ).rsample(torch.Size([num_X_samples]))

    def _sample_pstar_hnr(self, num_X_samples):
        # Sample from the approximate p*(x) within the bounding box.
        X_max = self.appx_normal.mu
        X_max = torch.maximum(self._eps_interior, torch.minimum(1 - self._eps_interior, X_max))
        num_dim = len(X_max.flatten())
        X = torch.tile(X_max, (num_X_samples, 1))

        # num_mcmc = num_dim * self.k_mcmc
        num_mcmc = self.k_mcmc

        sigma = self.appx_normal.sigma.detach().numpy().copy()
        # Lengthscale advice, pg. 3 of https://www.cs.cmu.edu/~epxing/Class/10708-14/scribe_notes/scribe_note_lecture17.pdf
        # eps = float(self.appx_normal.sigma.min())
        eps = np.prod(sigma) ** (1 / num_dim)
        print("EPS:", eps)
        max_iterations = 1000 * num_mcmc

        pmc = ParallelMCMCConvergence()
        for i_iter in range(max_iterations):
            i, X_1 = self._hnr_propose(X, eps, sigma=sigma)

            X[i] = X_1[i]
            if pmc.converged(X.detach().numpy()):
                break

        else:
            print(f"WARNING: MCMC did not converge in {max_iterations} iterations.")

        return X

    def _hnr_propose(self, X, eps, sigma):
        X_np = X.detach().numpy()
        u, llambda_minus, llambda_plus = find_perturbation_direction(
            X=X_np,
            num_tries=5,
            eps_bound=min(eps, float(self._eps_interior)) / 100,
            sigma=sigma,
        )

        # Make a 1D perturbation
        X_1 = torch.tensor(perturb_normal(X_np, u, eps, llambda_minus, llambda_plus))

        # Metropolis update
        p_0 = self.appx_normal.p_normal(X)
        p_1 = self.appx_normal.p_normal(X_1)
        p_change = torch.min(torch.tensor([1]), p_1 / p_0)
        i = np.where(p_0 == 0)[0]
        p_change[i] = 1
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
