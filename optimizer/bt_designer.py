import numpy as np
from botorch.optim import optimize_acqf

import common.all_bounds as all_bounds
from bo.acq_bt import AcqBT
from optimizer.sobol_designer import SobolDesigner


class BTDesigner:
    def __init__(
        self,
        policy,
        acq_fn,
        *,
        acq_kwargs=None,
        init_sobol=1,
        init_X_samples=False,
        sample_X_samples=False,
        opt_sequential=False,
        optimizer_options={"batch_limit": 10, "maxiter": 500}
    ):
        self._policy = policy
        self._acq_fn = acq_fn
        self._init_sobol = init_sobol
        self._sample_X_samples = sample_X_samples
        self._init_X_samples = init_X_samples
        self._opt_sequential = opt_sequential
        self._optimizer_options = optimizer_options
        self._acq_kwargs = acq_kwargs
        self._sobol = SobolDesigner(policy.clone())

    def __call__(self, data, num_arms):
        import warnings

        if len(data) < self._init_sobol:
            return self._sobol(data, num_arms)

        num_dim = self._policy.num_params()
        acqf = AcqBT(self._acq_fn, data, num_dim, self._acq_kwargs)
        if self._sample_X_samples:
            print("Sampling X_samples")
            X_samples = acqf.acq_function.X_samples
            assert len(X_samples) >= num_arms, (len(X_samples), num_arms)
            i = np.arange(len(X_samples))
            i = np.random.choice(i, size=(int(num_arms)), replace=False)
            X_cand = X_samples[i]
        else:
            warnings.simplefilter("ignore")
            if self._init_X_samples and hasattr(acqf.acq_function, "X_samples"):
                X = acqf.acq_function.X_samples
                batch_limit = 10
                i = np.random.choice(np.arange(len(X)), size=(num_arms * batch_limit,))
                # batch_size x q x num_dim
                batch_initial_conditions = X[i, :].reshape(batch_limit, num_arms, num_dim)
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
            x = (x.detach().numpy().flatten() - all_bounds.bt_low) / all_bounds.bt_width
            p = all_bounds.p_low + all_bounds.p_width * x
            policy.set_params(p)
            policies.append(policy)
        return policies
