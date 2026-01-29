from typing import Any

import torch

from acq.turbo_yubo.turbo_yubo_config import TurboYUBOConfig


class AcqTurboYUBO:
    def __init__(
        self,
        model,
        trman: Any,
        config: TurboYUBOConfig,
        obs_X: torch.Tensor,
        obs_Y_raw: torch.Tensor,
        max_candidates: int = 5000,
    ):
        assert model is not None
        assert trman is not None
        assert config is not None
        assert obs_X is not None
        assert obs_Y_raw is not None

        self.config = config
        self.model = model

        num_dim = obs_X.shape[-1]

        assert trman.num_dim == num_dim, (trman.num_dim, num_dim)
        self.state = trman

        self.num_candidates = min(100 * self.state.num_dim, max_candidates)
        self.device = obs_X.device
        self.dtype = obs_X.dtype

        self.X = obs_X
        self.Y = obs_Y_raw.detach()

        self.state.update_from_model(self.Y)

    def _x_center(self):
        if len(self.Y) == 0:
            return None
        best_idx = torch.argmax(self.Y).item()
        return self.X[best_idx : best_idx + 1, :]

    def _create_trust_region(self, x_center):
        if len(self.Y) == 0:
            return None, None
        if hasattr(self.model, "covar_module"):
            covar_module = self.model.covar_module
            if hasattr(covar_module, "base_kernel"):
                kernel = covar_module.base_kernel
            else:
                kernel = covar_module
        else:
            kernel = None
        lb, ub = self.state.create_trust_region(x_center, kernel, len(self.Y))
        return lb, ub

    def _draw_initial(self, num_arms):
        return torch.tensor(
            self.config.candidate_initializer(
                num_arms,
                self.state.num_dim,
                seed=int(torch.randint(999999, (1,)).item()),
            ),
            dtype=self.dtype,
            device=self.device,
        )

    def draw(self, num_arms):
        x_center = self._x_center()
        self.state.pre_draw()
        if len(self.X) == 0:
            return self._draw_initial(num_arms)

        if hasattr(self.model, "set_x_center"):
            self.model.set_x_center(x_center)

        lb, ub = self._create_trust_region(x_center)
        if lb is None or ub is None:
            return self._draw_initial(num_arms)
        x_cand = self.config.candidate_sampler(
            x_center, lb, ub, self.num_candidates, self.device, self.dtype
        )
        x_target = self.config.candidate_selector(self.X, self.model, x_cand, num_arms)
        x_arm = self.config.targeter(x_center, x_target)

        return x_arm
