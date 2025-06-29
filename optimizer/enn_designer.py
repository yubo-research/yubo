import torch

import acq.acq_util as acq_util
import acq.fit_gp as fit_gp
from acq.acq_enn import AcqENN, ENNConfig

_acq_enn = None


class ENNDesigner:
    def __init__(self, policy, enn_config: ENNConfig, num_keep: int = None, keep_style: str = None, warm_starting=False):
        self._policy = policy
        self._enn_config = enn_config
        self._num_keep = num_keep
        self._keep_style = keep_style
        self._warm_starting = warm_starting

        self._i_data_last = 0
        self._dtype = torch.double
        self._device = torch.empty(size=(1,)).device

    def __call__(self, data, num_arms):
        data = acq_util.keep_data(data, self._keep_style, self._num_keep)

        if len(data) > 0:
            Y, X = fit_gp.extract_X_Y(data[self._i_data_last :], self._dtype, self._device)
            if self._warm_starting:
                self._i_data_last = len(data)
        else:
            X = torch.empty(size=(0, self._policy.num_params()))
            Y = torch.empty(size=(0, 1))

        global _acq_enn

        if not self._warm_starting or _acq_enn is None:
            _acq_enn = AcqENN(self._policy.num_params(), self._enn_config)

        _acq_enn.add(X, Y)
        X_a = torch.as_tensor(_acq_enn.draw(num_arms).copy())
        return fit_gp.mk_policies(self._policy, X_a)
