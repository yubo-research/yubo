import numpy as np
from botorch.optim import optimize_acqf

import common.all_bounds as all_bounds
from bo.acq_bt import AcqBT
from optimizer.datum import Datum
from optimizer.sobol_designer import SobolDesigner
from optimizer.trajectories import Trajectory


class BTDesigner:
    def __init__(self, policy, acq_fn, *, acq_kwargs=None, init_sobol=0):
        self._policy = policy
        self._acq_fn = acq_fn
        self._init_sobol = init_sobol
        self._acq_kwargs = acq_kwargs
        self._sobol = SobolDesigner(policy.clone())

    def __call__(self, data, num_arms):
        import warnings

        if len(data) < self._init_sobol:
            # policy = self._policy.clone()
            # p = all_bounds.p_low + all_bounds.p_width * (np.ones(shape=(policy.num_params(),)) / 2)
            # p = all_bounds.p_low + all_bounds.p_width * (np.random.uniform(size=(policy.num_params(), num_arms)))
            return self._sobol(data, num_arms)

        data_use = data
        num_arms_use = num_arms
        x_extra = None
        if len(data) == 0:
            if num_arms == 1:
                return self._sobol(data, num_arms)
            else:
                p = self._policy.clone()
                x_extra = np.random.uniform(size=(p.num_params(),))

                p.set_params(all_bounds.p_low + all_bounds.p_width * x_extra)
                data_use = [Datum(p, Trajectory(0.0, None, None))]
                num_arms_use = num_arms - 1

        acqf = AcqBT(self._acq_fn, data_use, self._acq_kwargs)

        warnings.simplefilter("ignore")
        with warnings.catch_warnings():
            X_cand, _ = optimize_acqf(
                acq_function=acqf.acq_function,
                bounds=acqf.bounds,  # always [0,1]
                q=num_arms_use,
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
