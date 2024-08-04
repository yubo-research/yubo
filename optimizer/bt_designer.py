import numpy as np
import torch
from botorch.optim import optimize_acqf

import common.all_bounds as all_bounds
from acq.acq_bt import AcqBT
from optimizer.sobol_designer import SobolDesigner


class BTDesigner:
    def __init__(
        self,
        policy,
        acq_fn,
        *,
        acq_kwargs=None,
        init_sobol=1,
        init_X_samples=True,
        opt_sequential=False,  # greed q, not joint q
        num_keep=None,
        use_vanilla=False,
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
        self._use_vanilla = use_vanilla
        self._optimizer_options = optimizer_options
        self._acq_kwargs = acq_kwargs
        self.device = torch.device(device)
        self.dtype = dtype

    def __repr__(self):
        return f"{self.__class__.__name__} {self._acq_fn}"

    def _batch_initial_conditions(self, data, num_arms, acqf):
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

    def __call__(self, data, num_arms):
        import warnings

        if len(data) < self._init_sobol:
            sobol = SobolDesigner(self._policy.clone())
            ret = sobol(data, num_arms)
            self.fig_last_acqf = "sobol"
            self.fig_last_arms = sobol.fig_last_arms
            return ret

        num_dim = self._policy.num_params()
        acqf = AcqBT(
            self._acq_fn,
            data,
            num_dim,
            self._acq_kwargs,
            device=self.device,
            dtype=self.dtype,
            num_keep=self._num_keep,
            use_vanilla=self._use_vanilla,
        )
        if hasattr(acqf.acq_function, "draw"):
            # print (f"Draw from {acqf.acq_function.__class__.__name__}")
            X_cand = acqf.acq_function.draw(num_arms)
        else:
            warnings.simplefilter("ignore")
            if self._init_X_samples and hasattr(acqf.acq_function, "X_samples"):
                batch_initial_conditions = self._batch_initial_conditions(data, num_arms, acqf)
                batch_initial_conditions = batch_initial_conditions.type(self.dtype).to(self.device)
            else:
                batch_initial_conditions = None

            with warnings.catch_warnings():
                X_cand, _ = optimize_acqf(
                    acq_function=acqf.acq_function,
                    bounds=acqf.bounds,  # always [0,1]**num_dim
                    q=num_arms,
                    num_restarts=10,
                    raw_samples=10,
                    options=self._optimizer_options,
                    batch_initial_conditions=batch_initial_conditions,
                    sequential=self._opt_sequential,
                )

        self.fig_last_acqf = acqf
        self.fig_last_arms = X_cand

        policies = []
        for x in X_cand:
            policy = self._policy.clone()
            x = (x.detach().cpu().numpy().flatten() - all_bounds.bt_low) / all_bounds.bt_width
            p = all_bounds.p_low + all_bounds.p_width * x
            policy.set_params(p)
            policies.append(policy)
        return policies
