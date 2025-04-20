import torch

import acq.acq_util as acq_util
import acq.fit_gp as fit_gp
from acq.acq_enn import AcqENN, ENNConfig


class ENNDesigner:
    def __init__(self, policy, enn_config: ENNConfig, num_keep: int = None, keep_style: str = None):
        self._policy = policy
        self._enn_config = enn_config
        self._dtype = torch.double
        self._device = torch.empty(size=(1,)).device
        self._num_keep = num_keep
        self._keep_style = keep_style

    def __call__(self, data, num_arms):
        data = acq_util.keep_data(data, self._keep_style, self._num_keep)

        if len(data) > 0:
            Y, X = fit_gp.extract_X_Y(data, self._dtype, self._device)
            # TODO: Do we need to standardize?
            Y = fit_gp.standardize_torch(Y)
        else:
            X = torch.empty(size=(0, self._policy.num_params()))
            Y = torch.empty(size=(0, 1))
        enn = AcqENN(X, Y, self._enn_config)
        X_a = torch.as_tensor(enn.draw(num_arms).copy())
        return fit_gp.mk_policies(self._policy, X_a)
