import torch
from botorch.models import SingleTaskGP

import acq.fit_gp as fit_gp
from acq.acq_util import find_max, keep_best, keep_some


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
        model_type,
    ):
        # All BoTorch stuff is coded to bounds of [0,1]!
        self.bounds = torch.tensor([[0.0] * num_dim, [1.0] * num_dim], device=device, dtype=dtype)
        self._rebounds = None

        if len(data) == 0:
            X = torch.empty(size=(0, num_dim), dtype=dtype, device=device)
            Y = torch.empty(size=(0, 1), dtype=dtype, device=device)
        else:
            Y, X = fit_gp.extract_X_Y(data, dtype, device)
            if num_keep is not None and num_keep < len(X):
                if keep_style == "some":
                    i = keep_some(Y.squeeze(), num_keep)
                elif keep_style == "best":
                    i = keep_best(Y.squeeze(), num_keep)
                else:
                    assert False, keep_style
                Y = Y[i, :]
                X = X[i, :]
        gp = fit_gp.fit_gp_XY(X, Y, model_type=model_type)

        if not acq_kwargs:
            kwargs = {}
        else:
            kwargs = dict(acq_kwargs)
        if "X_max" in kwargs:
            kwargs["X_max"] = find_max(gp, self.bounds)
        if "best_f" in kwargs:
            kwargs["best_f"] = gp(find_max(gp, self.bounds)).mean
        if "X_baseline" in kwargs:
            kwargs["X_baseline"] = X
        if "candidate_set" in kwargs:
            kwargs["candidate_set"] = torch.rand(1000, num_dim)
        if "Y_max" in kwargs:
            kwargs["Y_max"] = gp(find_max(gp, self.bounds)).mean

        self.acq_function = acq_factory(gp, **kwargs)

    def __call__(self, policy):
        assert False, "This is never called"
        X = torch.atleast_2d(fit_gp.mk_x(policy)).unsqueeze(0)
        return self.acq_function(X).squeeze().item()
