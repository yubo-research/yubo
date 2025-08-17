import torch

import acq.acq_util as acq_util
import acq.fit_gp as fit_gp
from acq.acq_turbo_yubo import AcqTurboYUBO, TurboYUBOConfig, TurboYUBOState


class TurboYUBODesigner:
    def __init__(self, policy, num_keep: int = None, keep_style: str = None, raasp: bool = True):
        assert False, "Not ready for use"
        self._policy = policy
        self._num_keep = num_keep
        self._keep_style = keep_style
        self._i_data_last = 0
        self._X_train = torch.empty(size=(0, self._policy.num_params()))
        self._Y_train = torch.empty(size=(0, 1))
        self._turbo_yubo_state = None
        self._raasp = raasp

        self._dtype = torch.double
        self._device = torch.empty(size=(1,)).device

    def __call__(self, data, num_arms):
        if self._keep_style != "lap":
            data = acq_util.keep_data(data, self._keep_style, self._num_keep)
        else:
            data_use = data[self._i_data_last :]
            self._i_data_last = len(data)
            data = data_use

        if len(data) > 0:
            Y, X = fit_gp.extract_X_Y(data, self._dtype, self._device)

            self._X_train = torch.cat([self._X_train, X])
            self._Y_train = torch.cat([self._Y_train, Y])
            if self._keep_style == "lap":
                X = self._X_train
                Y = self._Y_train
        else:
            X = torch.empty(size=(0, self._policy.num_params()))
            Y = torch.empty(size=(0, 1))

        model = fit_gp.fit_gp_XY(X, Y)

        if self._turbo_yubo_state is None:
            self._turbo_yubo_state = TurboYUBOState(num_dim=self._policy.num_params(), batch_size=num_arms)

        acq_turbo = AcqTurboYUBO(model=model, state=self._turbo_yubo_state, config=TurboYUBOConfig(raasp=self._raasp))
        X_a = acq_turbo.draw(num_arms)
        self._turbo_yubo_state = acq_turbo.get_state()

        if self._keep_style == "lap":
            self._X_train, self._Y_train = acq_util.keep_top_n(X, Y, self._num_keep)
            self._X_train = torch.as_tensor(self._X_train, dtype=self._dtype, device=self._device)
            self._Y_train = torch.as_tensor(self._Y_train, dtype=self._dtype, device=self._device)

        return fit_gp.mk_policies(self._policy, X_a)
