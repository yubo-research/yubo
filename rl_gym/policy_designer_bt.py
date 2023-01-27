import time

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
        if not isinstance(self._acq_fn, str):
            self._md_best = self._acq(self._policy_best)
        else:
            self._md_best = None
            if self._acq_fn == "sobol":
                self._sobol = SobolDesigner(self._policy_best.num_params())

    def _acq(self, policy):
        return self._acq_fn(policy)

    def _calc_trust(self, policy):
        return self._trust_distance_fn(self._datum_trusted, policy)

    def _clamp_params(self, params):
        return np.clip(params, -1, 1)

    def design_sobol(self):
        p = self._sobol.get()
        self._policy_best.set_params(p)
        return np.array([0, 0])

    def design_dumb(self, eps):
        p = self._policy_best.get_params()
        p = self._clamp_params(p + eps * np.random.normal(size=p.shape))
        self._policy_best.set_params(p)
        return np.array([0, 0])

    def design_random(self):
        p = self._policy_best.get_params()
        p = np.random.uniform(-1, 1, size=p.shape)
        self._policy_best.set_params(p)
        return np.array([0, 0])

    def design(self):
        if self._acq_fn == "dumb":
            return self.design_dumb(0.1)
        elif self._acq_fn == "sobol":
            return self.design_sobol()
        elif self._acq_fn == "random":
            return self.design_random()

        import warnings

        times = []
        t0 = time.time_ns()
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
        tf = time.time_ns()
        times.append(tf - t0)

        self._policy_best.set_params(X_cand.detach().numpy().flatten())

        return np.array(times)

    def get_policy(self):
        return self._policy_best
