import torch

from acq.turbo_yubo.turbo_yubo_config import TurboYUBOConfig


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
        best_idx = torch.argmax(self.Y).item()
        x_center = self.X[best_idx : best_idx + 1, :]
        x_cand = self.config.raasp(x_center, lb, ub, num_candidates, self.device, self.dtype)

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
