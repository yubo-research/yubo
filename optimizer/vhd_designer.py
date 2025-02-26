import torch

import acq.fit_gp as fit_gp
from acq.acq_vhd import AcqVHD


class VHDDesigner:
    def __init__(self, policy, k):
        self._policy = policy
        self._k = k
        self._dtype = torch.double
        self._device = torch.empty(size=(1,)).device

    def __call__(self, data, num_arms):
        # TODO: Permanently discard worst samples from data. Only keep best B, where B ~ 1000
        if len(data) > 0:
            Y, X = fit_gp.extract_X_Y(data, self._dtype, self._device)
        else:
            X = torch.empty(size=(0, self._policy.num_params()))
            Y = torch.empty(size=(0, 1))
        vhd = AcqVHD(X, Y, k=self._k)
        X_a = torch.as_tensor(vhd.draw(num_arms))
        return fit_gp.mk_policies(self._policy, X_a)
