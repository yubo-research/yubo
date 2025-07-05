from dataclasses import dataclass

import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_samples

from sampling.lhd import latin_hypercube_design
from sampling.sampling_util import raasp


@dataclass
class TurboYUBOConfig:
    raasp: bool = True
    lhd: bool = False


@dataclass
class TurboYUBOState:
    num_dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    restart_triggered: bool = False
    prev_y_length: int = 0

    def __post_init__(self):
        self.failure_tolerance = np.ceil(max([4.0 / self.batch_size, float(self.num_dim) / self.batch_size]))

    def update_state(self, Y_next):
        if max(Y_next) > self.best_value + 1e-3 * np.fabs(self.best_value):
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter == self.success_tolerance:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:
            self.length /= 2.0
            self.failure_counter = 0

        self.best_value = max(self.best_value, max(Y_next).item())
        if self.length < self.length_min:
            self.restart_triggered = True
        return self

    def update_from_model(self, Y):
        if len(Y) > self.prev_y_length:
            new_Y = Y[self.prev_y_length :]
            self.update_state(new_Y)
            self.prev_y_length = len(Y)

    def restart(self):
        self.length = self.length_min
        self.success_counter = 0
        self.failure_counter = 0
        self.restart_triggered = False


class AcqTurboYUBO:
    def __init__(self, model, state=None, config=None):
        assert model is not None, "Model must be provided to AcqTurbo constructor"
        self.model = model
        self.config = config or TurboYUBOConfig()

        X_0 = model.train_inputs[0].detach()
        num_dim = X_0.shape[-1]
        batch_size = 1

        if state is None:
            state = TurboYUBOState(num_dim=num_dim, batch_size=batch_size)
        else:
            if state.num_dim != num_dim:
                raise ValueError(f"State dimension ({state.num_dim}) must match model dimension ({num_dim})")

        self.state = state

        self.n_candidates = min(100 * self.state.num_dim, 5000)
        self.device = X_0.device
        self.dtype = X_0.dtype

        self.X = model.train_inputs[0].detach()
        self.Y = model.train_targets.detach()

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
        if hasattr(kernel, "lengthscale"):
            weights = kernel.lengthscale.cpu().detach().numpy().ravel()
            weights = weights / weights.mean()
            weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))
        else:
            weights = np.ones(self.state.num_dim)
        lb = np.clip(x_center.cpu().numpy() - weights * self.state.length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center.cpu().numpy() + weights * self.state.length / 2.0, 0.0, 1.0)
        return lb, ub

    def _sample_candidates(self, lb, ub, n_candidates):
        if self.config.raasp:
            best_idx = torch.argmax(self.Y).item()
            x_center = self.X[best_idx : best_idx + 1, :]
            candidates = raasp(x_center, lb, ub, n_candidates, self.device, self.dtype)
        else:
            bounds = np.array([lb, ub])
            bounds = torch.tensor(bounds, dtype=self.dtype, device=self.device)
            candidates = draw_sobol_samples(bounds=bounds, n=n_candidates, q=1).squeeze(1)

        return candidates

    def _thompson_sample(self, x_cand, num_arms):
        if len(self.X) == 0:
            indices = torch.randperm(len(x_cand))[:num_arms]
            return x_cand[indices]
        with torch.no_grad():
            posterior = self.model.posterior(x_cand)
            samples = posterior.rsample(sample_shape=torch.Size([num_arms]))
            best_indices = torch.argmax(samples, dim=1)
            x_arm = x_cand[best_indices]
            while len(torch.unique(x_arm, dim=0)) < len(x_arm):
                samples = posterior.rsample(sample_shape=torch.Size([num_arms]))
                best_indices = torch.argmax(samples, dim=1)
                x_arm = x_cand[best_indices]
            return x_arm

    def _draw_uniform(self, num_arms):
        if self.config.lhd:
            lhd_samples = latin_hypercube_design(num_arms, self.state.num_dim, seed=int(torch.randint(999999, (1,)).item()))
            x_arm = torch.tensor(lhd_samples, dtype=self.dtype, device=self.device)
        else:
            x_arm = draw_sobol_samples(
                bounds=torch.tensor([[0.0] * self.state.num_dim, [1.0] * self.state.num_dim], dtype=self.dtype, device=self.device),
                n=num_arms,
                q=1,
            ).squeeze(1)
        return x_arm

    def draw(self, num_arms):
        if self.state.restart_triggered:
            self.state.restart()
        if len(self.X) == 0:
            return self._draw_uniform(num_arms)
        lb, ub = self._create_trust_region()
        if lb is None or ub is None:
            return self._draw_uniform(num_arms)
        x_cand = self._sample_candidates(lb, ub, self.n_candidates)
        x_arm = self._thompson_sample(x_cand, num_arms)
        return x_arm
