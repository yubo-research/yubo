import numpy as np
import torch
from torch.quasirandom import SobolEngine

from acq.turbo_yubo_config import TurboYUBOConfig
from sampling.sampling_util import raasp_turbo_np

"""
677bd902b83cf0b0f01a7523676c9b385ce7bf4f
  Designer "turbo-yubo" matches the reference designed, turbo-1, in y_max and proposal time on tlunar:fn.  See notes/turbo_yubo.png.
"""


class AcqTurboYUBO:
    def __init__(self, model, state=None, config=None, obs_X=None, obs_Y_raw=None):
        self.config = config or TurboYUBOConfig()
        if model is None and obs_X is not None and obs_Y_raw is not None:
            self.model = self.config.model_factory(train_x=obs_X, train_y=obs_Y_raw)
        else:
            assert model is not None, "Model must be provided to AcqTurbo constructor"
            self.model = model

        X_0 = (obs_X if obs_X is not None else self.model.train_inputs[0]).detach()
        num_dim = X_0.shape[-1]

        if state.num_dim != num_dim:
            raise ValueError(f"State dimension ({state.num_dim}) must match model dimension ({num_dim})")

        self.state = state

        self.num_candidates = min(100 * self.state.num_dim, 5000)
        self.device = X_0.device
        self.dtype = X_0.dtype

        self.X = X_0
        self.Y = (obs_Y_raw if obs_Y_raw is not None else self.model.train_targets).detach()

        self.state.update_from_model(self.Y)

    def get_state(self):
        return self.state

    def _create_trust_region(self):
        if len(self.Y) == 0:
            return None, None
        best_idx = torch.argmax(self.Y).item()
        x_center = self.X[best_idx : best_idx + 1, :]
        covar_module = self.model.covar_module
        if hasattr(covar_module, "base_kernel"):
            kernel = covar_module.base_kernel
        else:
            kernel = covar_module
        lb, ub = self.state.create_trust_region(x_center, kernel)
        return lb, ub

    def _sample_candidates(self, lb, ub, num_candidates):
        if self.config.raasp:
            best_idx = torch.argmax(self.Y).item()
            x_center = self.X[best_idx : best_idx + 1, :]
            # Use numpy-based RAASP matching turbo-1 style to minimize device churn
            x_cand = raasp_turbo_np(x_center, lb, ub, num_candidates, self.device, self.dtype)
        else:
            # Match turbo_1 SobolEngine + mask behavior
            best_idx = torch.argmax(self.Y).item()
            x_center = self.X[best_idx : best_idx + 1, :].cpu().numpy()
            lb_np = np.asarray(lb)
            ub_np = np.asarray(ub)
            sobol = SobolEngine(self.state.num_dim, scramble=True, seed=np.random.randint(int(1e6)))
            pert = sobol.draw(num_candidates).to(dtype=self.dtype, device=self.device).cpu().detach().numpy()
            pert = lb_np + (ub_np - lb_np) * pert

            prob_perturb = min(20.0 / self.state.num_dim, 1.0)
            mask = np.random.rand(num_candidates, self.state.num_dim) <= prob_perturb
            ind = np.where(np.sum(mask, axis=1) == 0)[0]
            if len(ind) > 0:
                mask[ind, np.random.randint(0, self.state.num_dim - 1, size=len(ind))] = 1

            X_cand = x_center.copy() * np.ones((num_candidates, self.state.num_dim))
            X_cand[mask] = pert[mask]
            x_cand = torch.tensor(X_cand, dtype=self.dtype, device=self.device)

        return x_cand

    def _thompson_sample(self, x_cand, num_arms):
        if len(self.X) == 0:
            indices = torch.randperm(len(x_cand))[:num_arms]
            return x_cand[indices]
        with torch.no_grad():
            posterior = self.model.posterior(x_cand)
            samples = posterior.sample(sample_shape=torch.Size([num_arms])).squeeze(-1)
            if samples.dim() == 1:
                samples = samples.unsqueeze(0)
            y_cand = samples.t().contiguous()
            # Greedy unique selection across arms (maximize)
            chosen = []
            for i in range(num_arms):
                indbest = torch.argmax(y_cand[:, i]).item()
                chosen.append(indbest)
                y_cand[indbest, :] = -float("inf")
            chosen = torch.tensor(chosen, device=x_cand.device)
            return x_cand[chosen]

    def _draw_initial(self, num_arms):
        return torch.tensor(
            self.config.initializer(num_arms, self.state.num_dim, seed=int(torch.randint(999999, (1,)).item())),
            dtype=self.dtype,
            device=self.device,
        )

    def draw(self, num_arms):
        self.state.pre_draw()
        if len(self.X) == 0:
            return self._draw_initial(num_arms)

        lb, ub = self._create_trust_region()
        if lb is None or ub is None:
            return self._draw_initial(num_arms)
        x_cand = self._sample_candidates(lb, ub, self.num_candidates)
        x_arm = self._thompson_sample(x_cand, num_arms)
        return x_arm
