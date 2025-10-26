import torch


class FullSpaceTR:
    def __init__(self, num_dim: int, num_arms: int):
        self.num_dim = num_dim
        self.num_arms = num_arms

    def update_from_model(self, Y):
        pass

    def pre_draw(self):
        pass

    def create_trust_region(self, x_center: torch.Tensor, kernel: torch.Tensor, num_obs: int):
        return torch.zeros(self.num_dim), torch.ones(self.num_dim)


class PartialTargeter:
    def __init__(self, alpha: float):
        self._alpha = alpha

    def __call__(self, x_center: torch.Tensor, x_target: torch.Tensor):
        return x_center + self._alpha * (x_target - x_center)
