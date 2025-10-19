import torch

import acq.acq_util as acq_util
import acq.fit_gp as fit_gp
from acq.acq_turbo_yubo import AcqTurboYUBO, TurboYUBOConfig, TurboYUBORestartError, TurboYUBOState
from acq.fit_gp_turbo import train_gp as turbo_train_gp


class TurboYUBODesigner:
    def __init__(self, policy, num_keep: int = None, keep_style: str = None, raasp: bool = True, tr: bool = True):
        self._policy = policy
        self._num_keep = num_keep
        self._keep_style = keep_style
        self._i_data_0 = 0
        self._X_train = torch.empty(size=(0, self._policy.num_params()))
        self._Y_train = torch.empty(size=(0, 1))
        self._turbo_yubo_state = None
        self._raasp = raasp
        self._tr = tr
        self._dtype = torch.double
        self._device = torch.empty(size=(1,)).device

    def __call__(self, data, num_arms):
        if self._keep_style != "lap":
            data = acq_util.keep_data(data, self._keep_style, self._num_keep)
        else:
            assert False, "lap not supported"

        for _ in range(2):
            try:
                return self._run_opt(data[self._i_data_0 :], num_arms)
            except TurboYUBORestartError:
                self._i_data_0 = len(data)
        raise RuntimeError("Restarted twice")

    def _run_opt(self, data, num_arms):
        if len(data) > 0:
            Y, X = fit_gp.extract_X_Y(data, self._dtype, self._device)

            self._X_train = torch.cat([self._X_train, X])
            self._Y_train = torch.cat([self._Y_train, Y])
        else:
            X = torch.empty(size=(0, self._policy.num_params()))
            Y = torch.empty(size=(0, 1))

        # Build a TuRBO-style GP on standardized targets (median/STD) in [0,1]^d
        if len(X) == 0:
            # Avoid constructing a GP when there is no data; mirror turbo-1 behavior
            class _EmptyModel:
                def __init__(self, X):
                    self.train_inputs = (X,)
                    self.train_targets = torch.empty(size=(0,))
                    self.covar_module = None

                def posterior(self, X):
                    raise RuntimeError("Posterior requested with no data")

            model = _EmptyModel(X)
            y_raw = Y.squeeze(-1)
        else:
            y_raw = Y.squeeze(-1)
            mu = torch.median(y_raw)
            sigma = y_raw.std()
            if sigma < 1e-6:
                sigma = torch.tensor(1.0, dtype=y_raw.dtype, device=y_raw.device)
            y_std = (y_raw - mu) / sigma

            gp = turbo_train_gp(train_x=X, train_y=y_std, use_ard=True, num_steps=50, hypers={})

            class _TurboPosteriorModel:
                def __init__(self, gp):
                    self._gp = gp
                    self.train_inputs = gp.train_inputs
                    # Keep standardized targets here; raw targets are supplied separately to AcqTurboYUBO
                    self.train_targets = gp.train_targets
                    self.covar_module = gp.covar_module

                def posterior(self, X):
                    with torch.no_grad():
                        return self._gp.likelihood(self._gp(X))

            model = _TurboPosteriorModel(gp)

        if self._turbo_yubo_state is None:
            self._turbo_yubo_state = TurboYUBOState(num_dim=self._policy.num_params(), batch_size=num_arms)

        acq_turbo = AcqTurboYUBO(
            model=model,
            state=self._turbo_yubo_state,
            config=TurboYUBOConfig(raasp=self._raasp, tr=self._tr),
            obs_X=X,
            obs_Y_raw=y_raw,
        )
        X_a = acq_turbo.draw(num_arms)
        self._turbo_yubo_state = acq_turbo.get_state()

        if self._keep_style == "lap":
            self._X_train, self._Y_train = acq_util.keep_top_n(X, Y, self._num_keep)
            self._X_train = torch.as_tensor(self._X_train, dtype=self._dtype, device=self._device)
            self._Y_train = torch.as_tensor(self._Y_train, dtype=self._dtype, device=self._device)

        return fit_gp.mk_policies(self._policy, X_a)
