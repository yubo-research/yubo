import torch


class RBFKernel:
    def __init__(self, input_scale):
        self._input_scale = input_scale

    def __call__(self, distance):
        return torch.exp(-(distance**2) / (2 * self._input_scale))
