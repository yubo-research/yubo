import sys
import time
from dataclasses import dataclass

import numpy as np
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
    qMaxValueEntropy,
)
from botorch.acquisition.monte_carlo import (
    qNoisyExpectedImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)

from bo.acq_iopt import AcqIOpt
from bo.acq_min_dist import AcqMinDist
from bo.acq_mtav import AcqMTAV
from bo.acq_var import AcqVar

from .ax_designer import AxDesigner
from .bt_designer import BTDesigner
from .center_designer import CenterDesigner
from .datum import Datum
from .random_designer import RandomDesigner
from .sobol_designer import SobolDesigner
from .trajectories import collect_trajectory
from .turbo_designer import TuRBODesigner


@dataclass
class _TraceEntry:
    rreturn: float
    time_iteration_seconds: float


class Optimizer:
    def __init__(self, env_conf, policy, num_arms, cb_trace=None):
        self._env_conf = env_conf
        self._num_arms = num_arms

        self._data = []
        self._datum_best = None
        self._i_iter = 0
        self._center_designer = CenterDesigner(policy)

        init_ax_default = max(5, 2 * policy.num_params())
        default_num_X_samples = max(64, 10 * self._num_arms)

        self._designers = {
            "random": RandomDesigner(policy, init_center=False),
            "random_center": RandomDesigner(policy, init_center=True),
            "sobol": SobolDesigner(policy, init_center=False),
            "sobol_center": SobolDesigner(policy, init_center=True),
            "maximin": BTDesigner(policy, lambda m: AcqMinDist(m, toroidal=False)),
            "maximin-toroidal": BTDesigner(policy, lambda m: AcqMinDist(m, toroidal=True)),
            "variance": BTDesigner(policy, AcqVar),
            "iopt": BTDesigner(policy, AcqIOpt, init_sobol=0, init_center=False),
            "mcmc_ts": BTDesigner(
                policy,
                AcqMTAV,
                init_sobol=0,
                init_center=False,
                sample_X_samples=True,
                acq_kwargs={"ttype": None, "num_X_samples": default_num_X_samples},
            ),
            "mtav_ts": BTDesigner(policy, AcqMTAV, init_sobol=0, init_center=False, acq_kwargs={"ttype": "maxvar", "num_X_samples": default_num_X_samples}),
            "mtav_msts": BTDesigner(
                policy,
                AcqMTAV,
                init_sobol=0,
                init_center=False,
                acq_kwargs={"ttype": "msvar", "num_X_samples": default_num_X_samples},
            ),
            "mtav_ei": BTDesigner(policy, AcqMTAV, init_sobol=0, init_center=False, acq_kwargs={"ttype": "ei", "num_X_samples": default_num_X_samples}),
            "mtav_msei": BTDesigner(policy, AcqMTAV, init_sobol=0, init_center=False, acq_kwargs={"ttype": "msei", "num_X_samples": default_num_X_samples}),
            "mtav_ucb": BTDesigner(
                policy,
                AcqMTAV,
                init_X_samples=True,
                init_sobol=0,
                init_center=False,
                acq_kwargs={"ttype": "ucb", "beta_ucb": 1.0, "num_X_samples": default_num_X_samples},
            ),
            "mtav_msucb": BTDesigner(policy, AcqMTAV, init_sobol=0, init_center=False, acq_kwargs={"ttype": "msucb", "num_X_samples": default_num_X_samples}),
            "sr": BTDesigner(policy, qSimpleRegret),
            "ucb": BTDesigner(policy, qUpperConfidenceBound, acq_kwargs={"beta": 1}),
            "ei": BTDesigner(policy, qNoisyExpectedImprovement, acq_kwargs={"X_baseline": None}),
            "mes": BTDesigner(policy, qMaxValueEntropy, acq_kwargs={"candidate_set": None}),
            "sobol_mes": BTDesigner(policy, qMaxValueEntropy, init_sobol=init_ax_default, acq_kwargs={"candidate_set": None}),
            "gibbon": BTDesigner(policy, qLowerBoundMaxValueEntropy, opt_sequential=True, acq_kwargs={"candidate_set": None}),
            "sobol_gibbon": BTDesigner(policy, qLowerBoundMaxValueEntropy, init_sobol=init_ax_default, acq_kwargs={"candidate_set": None}),
            "ax": AxDesigner(policy),
            "turbo": TuRBODesigner(policy, num_init=init_ax_default),
            "sobol_ei": BTDesigner(policy, qNoisyExpectedImprovement, init_sobol=init_ax_default, acq_kwargs={"X_baseline": None}),
            "sobol_ucb": BTDesigner(policy, qUpperConfidenceBound, init_sobol=init_ax_default, acq_kwargs={"beta": 1}),
        }

        for beta in [0, 1, 2, 3]:
            self._designers[f"mtav_msts_beta={beta}"] = BTDesigner(
                policy, AcqMTAV, init_sobol=0, init_center=False, acq_kwargs={"ttype": "msvar", "num_X_samples": default_num_X_samples, "beta": beta}
            )

    def collect_trajectory(self, policy):
        return collect_trajectory(self._env_conf, policy, seed=self._env_conf.seed)

    def _iterate(self, designer, num_arms):
        t0 = time.time()
        policies = designer(self._data, num_arms)
        tf = time.time()
        data = []
        for policy in policies:
            traj = self.collect_trajectory(policy)
            data.append(Datum(policy, traj))
        return data, tf - t0

    def collect_trace(self, ttype, num_iterations):
        assert ttype in self._designers, f"Unknown optimizer type {ttype}"

        designer = self._designers[ttype]
        if hasattr(designer, "init_center"):
            init_center = designer.init_center()
        else:
            init_center = True

        trace = []
        for _ in range(num_iterations):
            best_in_batch = -1e99
            if init_center and self._i_iter == 0:
                if self._num_arms == 1:
                    data, d_time = self._iterate(self._center_designer, self._num_arms)
                else:
                    data_c, _ = self._iterate(self._center_designer, num_arms=1)
                    data, d_time = self._iterate(designer, self._num_arms)
                    data[np.random.choice(np.arange(self._num_arms))] = data_c[0]
            else:
                data, d_time = self._iterate(designer, self._num_arms)

            for datum in data:
                self._data.append(datum)
                best_in_batch = max(best_in_batch, datum.trajectory.rreturn)
                if self._datum_best is None or datum.trajectory.rreturn > self._datum_best.trajectory.rreturn:
                    self._datum_best = datum

            print(f"ITER: self._i_iter = {self._i_iter} d_time = {d_time:.2f} ret = {best_in_batch:.2f} ret_best = {self._datum_best.trajectory.rreturn:.2f}")
            sys.stdout.flush()
            trace.append(_TraceEntry(self._datum_best.trajectory.rreturn, d_time))
            self._i_iter += 1

        return trace
