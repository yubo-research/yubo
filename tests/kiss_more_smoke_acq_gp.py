from types import SimpleNamespace

import torch


class _AcqBTGP:
    def __call__(self, _x):
        return SimpleNamespace(mean=torch.tensor(0.0))


class _AcqDPPModel:
    def __init__(self):
        self.train_inputs = (torch.zeros(2, 3, dtype=torch.double),)
        self.likelihood = SimpleNamespace(noise=torch.tensor(1.0, dtype=torch.double))

    def eval(self):
        return None
