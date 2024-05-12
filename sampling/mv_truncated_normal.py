import torch

from third_party.torch_truncnorm.TruncatedNormal import TruncatedNormal


class MVTruncatedNormal:
    def __init__(self, loc, scale):
        self.num_dim = len(loc)
        self.dtype = loc.dtype
        self.device = loc.device

        self._tn = TruncatedNormal(
            loc=loc,
            scale=scale,
            a=0,
            b=1,
        )

    def log_prob(self, X):
        lp = self._tn.log_prob(X)
        if self.num_dim > 1:
            lp = lp.sum(dim=1)
        return lp

    def unnormed_prob(self, X):
        lp = self.log_prob(X)
        return torch.exp(lp)

    def rsample(self, sample_shape):
        return self._tn.rsample(sample_shape)
