import numpy as np
from botorch.optim import optimize_acqf

from bo.acq_bt import AcqBT


class BTDesigner:
    def __init__(self, policy, acq_fn, acq_kwargs=None):
        self._policy = policy
        self._acq_fn = acq_fn
        self._acq_kwargs = acq_kwargs

    def __call__(self, data):
        import warnings

        if len(data) == 0:
            policy = self._policy.clone()
            p = np.zeros(shape=(policy.num_params(),))
            policy.set_params(p)
            return policy

        acqf = AcqBT(self._acq_fn, data, self._acq_kwargs)

        warnings.simplefilter("ignore")
        with warnings.catch_warnings():
            X_cand, _ = optimize_acqf(
                acq_function=acqf.acq_function,
                bounds=acqf.bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
            )
        policy = self._policy.clone()
        policy.set_params(X_cand.detach().numpy().flatten())
        return policy
