import torch

import acq.fit_gp as fit_gp
from acq.acq_vhd import AcqVHD, VHDConfig
from optimizer.designer_asserts import assert_scalar_rreturn


class VHDDesigner:
    def __init__(self, policy, vhd_config: VHDConfig):
        self._policy = policy
        self._vhd_config = vhd_config
        self._dtype = torch.double
        self._device = torch.empty(size=(1,)).device

    def __call__(self, data, num_arms, *, telemetry=None):
        # TODO: Permanently discard worst samples from data. Only keep best B, where B ~ 1000
        # TODO: Or, permanently discard samples not in the B ~ 1000 nearest neighbors to x_max
        assert_scalar_rreturn(data)
        if len(data) > 0:
            Y, X = fit_gp.extract_X_Y(data, self._dtype, self._device)
            Y = fit_gp.standardize_torch(Y)
        else:
            X = torch.empty(size=(0, self._policy.num_params()))
            Y = torch.empty(size=(0, 1))
        vhd = AcqVHD(X, Y, self._vhd_config)
        X_a = torch.as_tensor(vhd.draw(num_arms))
        return fit_gp.mk_policies(self._policy, X_a)
