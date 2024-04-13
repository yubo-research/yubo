import numpy as np
import torch

from sampling.hnr import find_perturbation_direction, perturb_normal


class PStarSampler:
    def __init__(self, k_mcmc, model, X_max):
        self.model = model
        self.k_mcmc = k_mcmc
        self.X_max = X_max
        self._eps_interior = torch.tensor(1e-6)
        self._eps_min = 1e-8
        self._num_dim = X_max.shape[-1]

    def __call__(self, num_X_samples):
        with torch.inference_mode():
            return self._sample_pstar(num_X_samples)

    def _sample_pstar(self, num_X_samples):
        X_max = torch.maximum(self._eps_interior, torch.minimum(1 - self._eps_interior, self.X_max))
        X = torch.tile(X_max, (num_X_samples, 1))

        num_mcmc = self._num_dim * self.k_mcmc

        eps = 1
        eps_good = False
        num_changed = 0
        max_iterations = 10 * num_mcmc
        frac_changed = None
        for _ in range(max_iterations):
            i, X_1 = self._hnr_propose(X, eps)
            X[i] = X_1[i]
            frac_changed = (1.0 * i).mean().item()
            # print("FC:", eps, eps_good, frac_changed)
            if frac_changed > 0.50 - 1e-5:
                eps_good = True
            elif frac_changed < 0.40:
                eps_good = False

            if not eps_good:
                eps = max(self._eps_min, eps / np.sqrt(10.0))
            else:
                num_changed += 1
                if num_changed == num_mcmc:
                    break
        else:
            print(
                f"WARNING: Could not determine eps in {max_iterations} iterations. Was at eps = {eps} and num_changed = {num_changed}. Last frac_changed = {frac_changed}"
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

        # Metropolis update (w/coarse approx. of p_*(x))
        X_both = torch.cat((X, X_1), dim=0)
        mvn = self.model.posterior(X_both, observation_noise=True)
        Y_both = mvn.sample(torch.Size([1])).squeeze(-1).squeeze(0)
        Y = Y_both[: len(X)]
        Y_1 = Y_both[len(X) :]
        b_met_accept = (Y_1 > Y).flatten()

        return b_met_accept, X_1
