import time

import torch

import acq.fit_gp as fit_gp
from acq.acq_util import find_max, keep_best, keep_some, keep_trailing

_KEEP_STYLE_FNS = {
    "some": keep_some,
    "best": keep_best,
    "trailing": keep_trailing,
}


def _extract_and_trim(data, dtype, device, num_dim, num_keep, keep_style):
    if len(data) == 0:
        X = torch.empty(size=(0, num_dim), dtype=dtype, device=device)
        Y = torch.empty(size=(0, 1), dtype=dtype, device=device)
        return X, Y
    Y, X = fit_gp.extract_X_Y(data, dtype, device)
    if num_keep is not None and num_keep < len(X):
        keep_fn = _KEEP_STYLE_FNS.get(keep_style)
        assert keep_fn is not None, keep_style
        i = keep_fn(Y.squeeze(), num_keep)
        Y = Y[i, :]
        X = X[i, :]
    return X, Y


class AcqBT:
    def __init__(
        self,
        acq_factory,
        data,
        num_dim,
        acq_kwargs=None,
        *,
        device,
        dtype,
        num_keep,
        keep_style,
        model_spec,
        telemetry=None,
    ):
        self.bounds = torch.tensor([[0.0] * num_dim, [1.0] * num_dim], device=device, dtype=dtype)
        self._rebounds = None

        X, Y = _extract_and_trim(data, dtype, device, num_dim, num_keep, keep_style)

        t0 = time.perf_counter()
        gp = fit_gp.fit_gp_XY(X, Y, model_spec=model_spec)
        dt_fit = time.perf_counter() - t0
        if telemetry is not None:
            telemetry.set_dt_fit(dt_fit)
        self._gp = gp

        kwargs = dict(acq_kwargs) if acq_kwargs else {}
        self._populate_kwargs(kwargs, gp, X, num_dim)
        self.acq_function = acq_factory(gp, **kwargs)

    def _populate_kwargs(self, kwargs, gp, X, num_dim):
        kwarg_values = {
            "X_max": lambda: self.x_max(),
            "best_f": lambda: gp(find_max(gp, self.bounds)).mean,
            "X_baseline": lambda: X,
            "candidate_set": lambda: torch.rand(1000, num_dim),
            "Y_max": lambda: gp(find_max(gp, self.bounds)).mean,
        }
        for key, value_fn in kwarg_values.items():
            if key in kwargs:
                kwargs[key] = value_fn()

    def model(self):
        return self._gp

    def x_max(self):
        return find_max(self._gp, self.bounds)

    def __call__(self, policy):
        assert False, "This is never called"
        X = torch.atleast_2d(fit_gp.mk_x(policy)).unsqueeze(0)
        return self.acq_function(X).squeeze().item()
