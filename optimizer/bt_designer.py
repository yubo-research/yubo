import time

import numpy as np
import torch
from botorch.optim import optimize_acqf

import acq.fit_gp as fit_gp
from acq.acq_bt import AcqBT
from optimizer.designer_asserts import assert_scalar_rreturn
from optimizer.sobol_designer import SobolDesigner


class BTDesigner:
    def __init__(
        self,
        policy,
        acq_fn,
        *,
        num_restarts,
        raw_samples,
        start_at_max,
        acq_kwargs=None,
        init_sobol=1,
        init_X_samples=True,
        opt_sequential=False,  # greedy q, not joint q
        num_keep=None,
        keep_style=None,
        model_spec=None,
        optimizer_options={"batch_limit": 10, "maxiter": 1000},
        dtype=torch.double,
        device=None,
    ):
        if device is None:
            device = torch.empty(size=(1,)).device

        self._policy = policy
        self._acq_fn = acq_fn
        self._init_sobol = init_sobol
        self._init_X_samples = init_X_samples
        self._opt_sequential = opt_sequential
        self._num_keep = num_keep
        self._keep_style = keep_style
        self._model_spec = model_spec
        self._optimizer_options = optimizer_options
        self._num_restarts = num_restarts
        self._raw_samples = raw_samples
        self._start_at_max = start_at_max
        self._acq_kwargs = acq_kwargs
        self.device = torch.device(device)
        self.dtype = dtype
        self._datum_best = None
        self._y_est_best = None

    def __repr__(self):
        return f"{self.__class__.__name__} {self._acq_fn}"

    def best_datum(self):
        return self._datum_best

    def _initialize_at_X_samples(self, data, num_arms, acqf):
        # half from X_samples, half random
        num_dim = self._policy.num_params()
        batch_limit = self._optimizer_options["batch_limit"]
        X_0 = acqf.acq_function.X_samples
        sobol = SobolDesigner(self._policy.clone(), max_points=len(X_0))
        X_s = torch.stack([torch.tensor(x.get_params()) for x in sobol(None, len(X_0))])

        X = torch.cat((X_0, X_s), dim=0)

        i = np.random.choice(
            np.arange(len(X)),
            size=(num_arms * batch_limit,),
            replace=True,
        )
        # batch_size x q x num_dim
        return X[i, :].reshape(batch_limit, num_arms, num_dim)

    def _initialize_at_x_max(self, acq_bt, num_arms):
        batch_limit = self._optimizer_options["batch_limit"]
        return (
            torch.tile(acq_bt.x_max().unsqueeze(0), dims=(num_arms * batch_limit, 1, 1))
            .to(self.device)
            .to(self.dtype)
        )

    def __call__(self, data, num_arms, *, telemetry=None):
        import warnings

        if len(data) < self._init_sobol:
            sobol = SobolDesigner(self._policy.clone())
            ret = sobol(data, num_arms, telemetry=telemetry)
            self.fig_last_acqf = "sobol"
            self.fig_last_arms = sobol.fig_last_arms
            return ret

        assert_scalar_rreturn(data)

        num_dim = self._policy.num_params()
        acq_bt = AcqBT(
            self._acq_fn,
            data,
            num_dim,
            self._acq_kwargs,
            device=self.device,
            dtype=self.dtype,
            num_keep=self._num_keep,
            keep_style=self._keep_style,
            model_spec=self._model_spec,
            telemetry=telemetry,
        )
        Y, X = fit_gp.extract_X_Y(data, self.dtype, self.device)
        y_est = (
            fit_gp.estimate(acq_bt.model(), X).detach().cpu().numpy().astype(np.float64)
        )
        best_i = int(np.argmax(y_est))
        best_y = float(y_est[best_i])
        for i, d in enumerate(data):
            d.trajectory.rreturn_est = float(y_est[i])
        if self._y_est_best is None or best_y > float(self._y_est_best):
            self._y_est_best = best_y
            self._datum_best = data[best_i]
        t0 = time.perf_counter()
        if hasattr(acq_bt.acq_function, "draw"):
            # print (f"Draw from {acqf.acq_function.__class__.__name__}")
            X_cand = acq_bt.acq_function.draw(num_arms)
        else:
            if self._init_X_samples and hasattr(acq_bt.acq_function, "X_samples"):
                batch_initial_conditions = self._initialize_at_X_samples(
                    data, num_arms, acq_bt
                )
                batch_initial_conditions = batch_initial_conditions.type(self.dtype).to(
                    self.device
                )
            elif self._start_at_max:
                batch_initial_conditions = self._initialize_at_x_max(acq_bt, num_arms)
            else:
                batch_initial_conditions = None

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", message=".*Optimization failed.*")
                X_cand, _ = optimize_acqf(
                    acq_function=acq_bt.acq_function,
                    bounds=acq_bt.bounds,  # always [0,1]**num_dim
                    q=num_arms,
                    num_restarts=self._num_restarts,
                    raw_samples=self._raw_samples,
                    options=self._optimizer_options,
                    batch_initial_conditions=batch_initial_conditions,
                    sequential=self._opt_sequential,
                )

        dt_select = time.perf_counter() - t0
        if telemetry is not None:
            telemetry.set_dt_select(dt_select)

        self.fig_last_acqf = acq_bt
        self.fig_last_arms = X_cand

        return fit_gp.mk_policies(self._policy, X_cand)
