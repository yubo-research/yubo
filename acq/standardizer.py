import torch


class Standardizer:
    def __init__(self, Y_orig):
        stddim = -1 if Y_orig.dim() < 2 else -2
        Y_std = Y_orig.std(dim=stddim, keepdim=True)
        self.Y_std = Y_std.where(Y_std >= 1e-9, torch.full_like(Y_std, 1.0))
        self.Y_mu = Y_orig.mean(dim=stddim, keepdim=True)

    def __call__(self, Y_orig):
        return (Y_orig - self.Y_mu) / self.Y_std

    def undo(self, Y):
        return self.Y_mu + self.Y_std * Y
