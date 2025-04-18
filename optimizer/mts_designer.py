import torch

import acq.fit_gp as fit_gp
from acq.acq_mts import AcqMTS


class MTSDesigner:
    def __init__(self, policy):
        self._policy = policy
        self._dtype = torch.double
        self._device = torch.empty(size=(1,)).device

    def __call__(self, data, num_arms):
        if len(data) > 0:
            Y, X = fit_gp.extract_X_Y(data, self._dtype, self._device)
            Y = fit_gp.standardize_torch(Y)
        else:
            X = torch.empty(size=(0, self._policy.num_params())).to(self._device).to(self._dtype)
            Y = torch.empty(size=(0, 1)).to(self._device).to(self._dtype)

        gp = fit_gp.fit_gp_XY(X, Y)
        mts = AcqMTS(gp)
        X_a = torch.as_tensor(mts.draw(num_arms))
        return fit_gp.mk_policies(self._policy, X_a)
