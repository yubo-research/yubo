import sys
import time

from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.acquisition.monte_carlo import (
    qNoisyExpectedImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)

from bo.acq_ieig import AcqIEIG
from bo.acq_iopt import AcqIOpt
from bo.acq_min_dist import AcqMinDist
from bo.acq_noisy_max import AcqNoisyMax
from bo.acq_var import AcqVar

from .ax_designer import AxDesigner
from .bt_designer import BTDesigner
from .datum import Datum
from .random_designer import RandomDesigner
from .sobol_designer import SobolDesigner
from .trajectories import collect_trajectory


def _iOptFactory(model, acqf=None, X_baseline=None):
    if acqf == "ei":
        assert X_baseline is not None
        if len(X_baseline) > 0:
            acqf = qNoisyExpectedImprovement(model, X_baseline, prune_baseline=False)
        else:
            acqf = None
    elif acqf == "sr":
        acqf = qSimpleRegret(model)
    elif acqf == "nm":
        acqf = AcqNoisyMax(model)
    elif acqf == "mes":
        acqf = qMaxValueEntropy(model, candidate_set=X_baseline)
    else:
        assert acqf is None, acqf
    return AcqIOpt(model, acqf=acqf)


class Optimizer:
    def __init__(self, env_conf, policy, num_arms, cb_trace=None):
        self._env_conf = env_conf
        self._num_arms = num_arms
        self._cb_trace = cb_trace
        self._data = []
        self._datum_best = None
        self._designers = {
            "random": RandomDesigner(policy),
            "sobol": SobolDesigner(policy),
            "maximin": BTDesigner(policy, lambda m: AcqMinDist(m, toroidal=False)),
            "maximin-toroidal": BTDesigner(policy, lambda m: AcqMinDist(m, toroidal=True)),
            "variance": BTDesigner(policy, AcqVar),
            "iopt": BTDesigner(policy, _iOptFactory, init_sobol=0),
            "iopt_sr": BTDesigner(policy, _iOptFactory, init_sobol=0, acq_kwargs={"acqf": "sr"}),
            "iopt_nm": BTDesigner(policy, _iOptFactory, init_sobol=0, acq_kwargs={"acqf": "nm"}),
            "iopt_ei": BTDesigner(policy, _iOptFactory, init_sobol=0, acq_kwargs={"acqf": "ei", "X_baseline": None}),
            "iopt_mes": BTDesigner(policy, _iOptFactory, init_sobol=0, acq_kwargs={"acqf": "mes", "X_baseline": None}),
            "ieig": BTDesigner(policy, AcqIEIG, init_sobol=0),
            "ieig_f4": BTDesigner(policy, AcqIEIG, init_sobol=0, acq_kwargs={"num_fantasies": 3, "num_Y_samples": 32}),
            "sr": BTDesigner(policy, qSimpleRegret),
            "ei": BTDesigner(policy, qNoisyExpectedImprovement, acq_kwargs={"X_baseline": None}),
            "ucb": BTDesigner(policy, qUpperConfidenceBound, acq_kwargs={"beta": 1}),
            "ax": AxDesigner(policy),
            "sobol_ei": BTDesigner(policy, qNoisyExpectedImprovement, init_sobol=max(5, 2 * policy.num_params()), acq_kwargs={"X_baseline": None}),
            "sobol_ucb": BTDesigner(policy, qUpperConfidenceBound, init_sobol=max(5, 2 * policy.num_params()), acq_kwargs={"beta": 1}),
        }

    def _collect_trajectory(self, policy):
        return collect_trajectory(self._env_conf, policy, seed=self._env_conf.seed)

    def _iterate(self, designer):
        t0 = time.time()
        policies = designer(self._data, self._num_arms)
        tf = time.time()
        data = []
        for policy in policies:
            traj = self._collect_trajectory(policy)
            data.append(Datum(policy, traj))
        return data, tf - t0

    def collect_trace(self, ttype, num_iterations):
        assert ttype in self._designers, f"Unknown optimizer type {ttype}"

        designer = self._designers[ttype]
        trace = []
        for i_iter in range(num_iterations):
            best_in_batch = -1e99
            data, d_time = self._iterate(designer)
            for datum in data:
                self._data.append(datum)
                best_in_batch = max(best_in_batch, datum.trajectory.rreturn)
                if self._datum_best is None or datum.trajectory.rreturn > self._datum_best.trajectory.rreturn:
                    self._datum_best = datum

            if i_iter % 1 == 0:
                print(f"ITER: i_iter = {i_iter} d_time = {d_time:.2f} ret = {best_in_batch:.2f} ret_best = {self._datum_best.trajectory.rreturn:.2f}")
                sys.stdout.flush()
            trace.append(self._datum_best.trajectory.rreturn)
            if self._cb_trace:
                self._cb_trace(self._datum_best)

        return trace
