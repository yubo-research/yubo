from __future__ import annotations

from typing import Any

import numpy as np

from .turbo_enn_designer import TurboENNDesigner
from .turbo_enn_varentropy_config import TurboENNVarentropyDesignerConfig
from .turbo_mars_designer import _check_scalar_metrics, _init_mars_optimizer, _optimizer_config


class TurboENNVarentropyDesigner(TurboENNDesigner):
    def __init__(self, policy: Any, *, config: TurboENNVarentropyDesignerConfig) -> None:
        self._varentropy_config = config.enn
        super().__init__(
            policy,
            turbo_mode="turbo-enn",
            k=config.enn.k,
            num_init=config.num_init,
            num_keep=config.num_keep,
            num_fit_samples=1,
            num_fit_candidates=1,
            acq_type=config.acq_type,
            tr_type=config.tr_type,
            use_y_var=bool(config.enn.include_noise_in_sigma),
            num_candidates=config.num_candidates,
            candidate_rv=config.candidate_rv,
            use_python=False,
            index_driver=config.enn.index_driver,
        )

    def _make_config(self, num_init: int, num_metrics: int | None):
        _check_scalar_metrics(num_metrics, "TurboENNVarentropyDesigner")
        return _optimizer_config(
            self,
            num_init=num_init,
            surrogate=self._varentropy_config,
        )

    def _init_optimizer(self, data, num_arms):
        _init_mars_optimizer(self, data, num_arms)

    def _estimate_new_data_for_best(self, new_data, x, y_est_0):
        del x
        surrogate = getattr(getattr(self, "_turbo", None), "_surrogate", None)
        x_obs = getattr(getattr(self, "_turbo", None), "_x_obs", None)
        if surrogate is None or x_obs is None or not hasattr(surrogate, "predict_leave_one_out"):
            return y_est_0
        n_new = len(new_data)
        if n_new <= 0:
            return y_est_0
        try:
            x_unit = np.asarray(x_obs.view(), dtype=float)[-n_new:]
            loo = surrogate.predict_leave_one_out(x_unit).mu
        except (RuntimeError, ValueError, TypeError, AttributeError):
            return y_est_0
        loo = np.asarray(loo, dtype=np.float64)
        if loo.shape != (n_new, 1):
            return y_est_0
        return loo[:, 0]


__all__ = ["TurboENNVarentropyDesigner", "TurboENNVarentropyDesignerConfig"]
