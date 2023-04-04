from botorch.optim import optimize_acqf

import common.all_bounds as all_bounds
from bo.acq_bt import AcqBT
from optimizer.sobol_designer import SobolDesigner


class BTDesigner:
    def __init__(self, policy, acq_fn, *, acq_kwargs=None, init_sobol=1):
        self._policy = policy
        self._acq_fn = acq_fn
        self._init_sobol = init_sobol
        self._acq_kwargs = acq_kwargs
        self._sobol = SobolDesigner(policy.clone())

    def __call__(self, data, num_arms):
        import warnings

        if len(data) < self._init_sobol:
            return self._sobol(data, num_arms)

        num_dim = self._policy.num_params()
        acqf = AcqBT(self._acq_fn, data, num_dim, self._acq_kwargs)
        if hasattr(acqf.acq_function, "X_cand"):
            X_cand = acqf.acq_function.X_cand
        else:
            warnings.simplefilter("ignore")
            with warnings.catch_warnings():
                X_cand, _ = optimize_acqf(
                    acq_function=acqf.acq_function,
                    bounds=acqf.bounds,  # always [0,1]**num_dim
                    q=num_arms,
                    num_restarts=10,
                    raw_samples=512,
                    options={"batch_limit": 5, "maxiter": 200},
                )

        policies = []
        for x in X_cand:
            policy = self._policy.clone()
            x = (x.detach().numpy().flatten() - all_bounds.bt_low) / all_bounds.bt_width
            p = all_bounds.p_low + all_bounds.p_width * x
            policy.set_params(p)
            policies.append(policy)
        return policies
