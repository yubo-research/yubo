import sys
import time
from dataclasses import dataclass

from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
    qMaxValueEntropy,
)
from botorch.acquisition.monte_carlo import (
    qNoisyExpectedImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)

from bo.acq_des import AcqDES
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


@dataclass
class _TraceEntry:
    rreturn: float
    time_iteration_seconds: float


def _kwargs(**kwargs):
    return kwargs


class Optimizer:
    def __init__(self, env_conf, policy, num_arms, cb_trace=None):
        self._env_conf = env_conf
        self._num_arms = num_arms
        self._cb_trace = cb_trace
        self._data = []
        self._datum_best = None
        init_ax_default = max(5, 2 * policy.num_params())
        self._designers = {
            "random": RandomDesigner(policy),
            "sobol": SobolDesigner(policy),
            "maximin": BTDesigner(policy, lambda m: AcqMinDist(m, toroidal=False)),
            "maximin-toroidal": BTDesigner(policy, lambda m: AcqMinDist(m, toroidal=True)),
            "variance": BTDesigner(policy, AcqVar),
            "iopt": BTDesigner(policy, _iOptFactory, init_sobol=0),
            "des": BTDesigner(policy, AcqDES, init_sobol=0, init_X_samples=True),
            "ieig": BTDesigner(policy, AcqIEIG, init_sobol=0, init_X_samples=False),
            "ieig_ks": BTDesigner(
                policy,
                AcqIEIG,
                init_sobol=0,
                init_X_samples=True,
                acq_kwargs=_kwargs(
                    num_X_samples=128,
                    num_px_weights=1024,
                    num_px_mc=128,
                    num_mcmc=10,
                    p_all_type="all",
                    num_fantasies=0,
                    num_Y_samples=32,
                    num_noisy_maxes=0,
                    q_ts=None,
                    no_log=True,
                    fantasies_only=True,
                    use_des=False,
                    no_weights=False,
                ),
            ),
            "ieig_ks4": BTDesigner(
                policy,
                AcqIEIG,
                init_sobol=0,
                init_X_samples=True,
                acq_kwargs=_kwargs(
                    num_X_samples=128,
                    num_px_weights=1024,
                    num_px_mc=128,
                    num_mcmc=10,
                    p_all_type="all",
                    num_fantasies=0,
                    num_Y_samples=32,
                    num_noisy_maxes=0,
                    q_ts=None,
                    no_log=True,
                    fantasies_only=True,
                    use_des=False,
                    no_weights=True,
                ),
            ),
            "ieig_yl": BTDesigner(policy, AcqIEIG, init_sobol=0, init_X_samples=True, acq_kwargs={"no_log": False}),
            "ieig_ts": BTDesigner(
                policy,
                AcqIEIG,
                init_sobol=0,
                acq_kwargs={
                    "use_cem": False,
                    "q_ts": num_arms,
                    "num_px_samples": 256,
                },
            ),
            "sr": BTDesigner(policy, qSimpleRegret),
            "ei": BTDesigner(policy, qNoisyExpectedImprovement, acq_kwargs={"X_baseline": None}),
            "mes": BTDesigner(policy, qMaxValueEntropy, acq_kwargs={"candidate_set": None}),
            "sobol_mes": BTDesigner(policy, qMaxValueEntropy, init_sobol=init_ax_default, acq_kwargs={"candidate_set": None}),
            "gibbon": BTDesigner(policy, qLowerBoundMaxValueEntropy, acq_kwargs={"candidate_set": None}),
            "sobol_gibbon": BTDesigner(policy, qLowerBoundMaxValueEntropy, init_sobol=init_ax_default, acq_kwargs={"candidate_set": None}),
            "ucb": BTDesigner(policy, qUpperConfidenceBound, acq_kwargs={"beta": 1}),
            "ax": AxDesigner(policy),
            "sobol_ei": BTDesigner(policy, qNoisyExpectedImprovement, init_sobol=init_ax_default, acq_kwargs={"X_baseline": None}),
            "sobol_ucb": BTDesigner(policy, qUpperConfidenceBound, init_sobol=init_ax_default, acq_kwargs={"beta": 1}),
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
            trace.append(_TraceEntry(self._datum_best.trajectory.rreturn, d_time))
            if self._cb_trace:
                self._cb_trace(self._datum_best)

        return trace
