import time

import cma
import numpy as np

from rl_gym.sobol_designer import SobolDesigner


# actions are always in -1,1; collect_trajectory() rescales them.
# parameters are in [-1,1]
class PolicyDesigner:
    def __init__(self, trust_distance_fn, acq_fn, data, delta_tr, i_trusted=-1):
        self._trust_distance_fn = trust_distance_fn
        self._acq_fn = acq_fn
        self._data = data
        self._delta_tr = delta_tr
        self._datum_trusted = data[i_trusted]

        self._policy_best = self._datum_trusted.policy.clone()
        self._dist_to_trusted = self._calc_trust(self._policy_best)
        assert self._dist_to_trusted < 1e-9, self._dist_to_trusted
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

    def design(self, num_iterations):
        if self._acq_fn == "dumb":
            return self.design_dumb(0.1)
        elif self._acq_fn == "sobol":
            return self.design_sobol()

        import warnings

        warnings.simplefilter("ignore")
        policy_test = self._policy_best.clone()
        times = []
        es = cma.CMAEvolutionStrategy([0] * policy_test.num_params(), sigma0=0.1, inopts={"verbose": -1})
        for _ in range(num_iterations):
            ws = es.ask()
            phis = []
            for w in ws:  # TODO: Could we do parallel evaluation and save some calls to NumPy?
                policy_test.set_params(self._clamp_params(w))
                t0 = time.time_ns()
                md_test = self._acq(policy_test)
                tf = time.time_ns()
                times.append(tf - t0)
                dist_to_trusted = self._calc_trust(policy_test)
                # print ("CHECK:", eps, md_test, self._md_best, dist_to_trusted, self._delta_tr)
                if md_test > self._md_best and dist_to_trusted <= self._delta_tr:
                    self._md_best = md_test
                    self._dist_to_trusted = dist_to_trusted
                    self._policy_best = policy_test.clone()
                    # print (f"BEST: md = {self._md_best:.4f} tr = {self._dist_to_trusted:.4f} {np.abs(self._policy_best.get_params()).max()}")
                phi = md_test - 1000 * max(0, (dist_to_trusted - self._delta_tr) / self._delta_tr)
                phis.append(phi)
            phis = np.array(phis)
            with warnings.catch_warnings():
                es.tell(ws, -phis)
        return np.array(times)

    def min_dist(self):
        return self._md_best

    def trust(self):
        return self._dist_to_trusted

    def get_policy(self):
        return self._policy_best
