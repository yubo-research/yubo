import numpy as np
import torch


class _FigEnv:
    def step(self, x):
        return None, float(np.sum(x)), False, False


class _FigEnvConf:
    def make(self):
        return _FigEnv()


class _FigPost:
    def __init__(self, n):
        self.mean = torch.zeros((n, 1))
        self.variance = torch.ones((n, 1))
        self._n = n

    def sample(self, size):
        return torch.zeros(size + torch.Size([self._n]))
