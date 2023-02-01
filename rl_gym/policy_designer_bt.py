import numpy as np
from botorch.optim import optimize_acqf

from rl_gym.sobol_designer import SobolDesigner


# actions are always in -1,1; collect_trajectory() rescales them.
# parameters are in [-1,1]
class PolicyDesignerBT:
    def __init__(self, acq_fn, data, i_best=-1):
        self._acq_fn = acq_fn
        self._data = data
        self._policy_best = data[i_best].policy.clone()

    def design(self):
        if self._acq_fn == "rs":
            self._design_rs()
        elif isinstance(self._acq_fn, SobolDesigner):
            self._design_sobol()
        elif self._acq_fn == "random":
            self._design_random()
        else:
            self._design_bt()
        return self._policy_best

    def get_policy(self):
        return self._policy_best

    def _design_sobol(self):
        p = self._acq_fn.get()
        self._policy_best.set_params(p)

    def _design_rs(self, eps=0.1):
        p = self._policy_best.get_params()
        p = np.clip(
            p + eps * np.random.normal(size=p.shape),
            -1,
            1,
        )
        self._policy_best.set_params(p)

    def _design_random(self):
        p = self._policy_best.get_params()
        p = np.random.uniform(-1, 1, size=p.shape)
        self._policy_best.set_params(p)

    def _design_bt(self):
        import warnings

        warnings.simplefilter("ignore")
        with warnings.catch_warnings():
            X_cand, _ = optimize_acqf(
                acq_function=self._acq_fn.acq_function,
                bounds=self._acq_fn.bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
            )
        self._policy_best.set_params(X_cand.detach().numpy().flatten())
