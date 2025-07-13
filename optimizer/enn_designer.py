import numpy as np
import torch

import acq.acq_util as acq_util
import acq.fit_gp as fit_gp
from acq.acq_enn import AcqENN, ENNConfig


class ENNDesigner:
    def __init__(self, policy, enn_config: ENNConfig, num_keep: int = None, keep_style: str = None):
        self._policy = policy
        self._enn_config = enn_config
        self._num_keep = num_keep
        self._keep_style = keep_style
        self._i_data_last = 0
        self._X_train = torch.empty(size=(0, self._policy.num_params()))
        self._Y_train = torch.empty(size=(0, 1))
        self._D_train = None

        self._dtype = torch.double
        self._device = torch.empty(size=(1,)).device

    def __call__(self, data, num_arms):
        if self._keep_style != "lap":
            data = acq_util.keep_data(data, self._keep_style, self._num_keep)
        else:
            data_use = data[self._i_data_last :]
            self._i_data_last = len(data)
            data = data_use

        D = None
        if len(data) > 0:
            # states ~ num_states X num_steps
            # actions ~ num_actions X num_steps
            Y, X = fit_gp.extract_X_Y(data, self._dtype, self._device)

            self._X_train = torch.cat([self._X_train, X])
            self._Y_train = torch.cat([self._Y_train, Y])

            # TODO: Try other descriptors
            D = []
            for datum in data:
                s = datum.trajectory.states  # shape: (state_dim, num_steps)
                num_steps = s.shape[1]
                idxs = np.linspace(0, num_steps - 1, 10).round().astype(int)
                idxs = np.clip(idxs, 0, num_steps - 1)
                D.append(torch.as_tensor(s[:, idxs], dtype=self._dtype, device=self._device).flatten())
            D = torch.stack(D, dim=0)

            if self._D_train is None:
                self._D_train = torch.empty(size=(0, D.shape[1]))

            self._D_train = torch.cat([self._D_train, D], dim=0)

            if self._keep_style == "lap":
                X = self._X_train
                Y = self._Y_train
                D = self._D_train
        else:
            X = torch.empty(size=(0, self._policy.num_params()))
            Y = torch.empty(size=(0, 1))

        acq_enn = AcqENN(self._policy.num_params(), self._enn_config)

        acq_enn.add(X, Y, D)

        X_a = torch.as_tensor(acq_enn.draw(num_arms).copy())

        if self._keep_style == "lap":
            self._X_train, self._Y_train = acq_enn.keep_top_n(self._num_keep)
            self._X_train = torch.as_tensor(self._X_train, dtype=self._dtype, device=self._device)
            self._Y_train = torch.as_tensor(self._Y_train, dtype=self._dtype, device=self._device)

        return fit_gp.mk_policies(self._policy, X_a)
