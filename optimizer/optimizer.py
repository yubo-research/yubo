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

from bo.acq_dpp import AcqDPP
from bo.acq_iopt import AcqIOpt
from bo.acq_min_dist import AcqMinDist
from bo.acq_mtv import AcqMTV
from bo.acq_ts import AcqTS

# from bo.acq_var import AcqVar
from .ax_designer import AxDesigner
from .bt_designer import BTDesigner
from .center_designer import CenterDesigner
from .cma_designer import CMADesigner
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

        print(f"PROBLEM: env = {env_conf.env_name} num_params = {policy.num_params()}")
        init_ax_default = max(5, 2 * policy.num_params())
        default_num_X_samples = max(64, 10 * self._num_arms)
        # default_num_Y_samples = 512

        self._designers = {
            "cma": CMADesigner(policy),
            "random": RandomDesigner(policy, init_center=False),
            "random_c": RandomDesigner(policy, init_center=True),
            "sobol": SobolDesigner(policy, init_center=False),
            "sobol_c": SobolDesigner(policy, init_center=True),
            "maximin": BTDesigner(policy, lambda m: AcqMinDist(m, toroidal=False)),
            "maximin-toroidal": BTDesigner(policy, lambda m: AcqMinDist(m, toroidal=True)),
            # FIXME: "variance": BTDesigner(policy, AcqVar),
            "dpp": BTDesigner(policy, AcqDPP, init_sobol=1, init_center=False, acq_kwargs={"num_X_samples": default_num_X_samples}),
            "dpp_c": BTDesigner(policy, AcqDPP, init_sobol=1, init_center=True, acq_kwargs={"num_X_samples": default_num_X_samples}),
            "iopt": BTDesigner(policy, AcqIOpt, init_sobol=0, init_center=False),
            "mtv": BTDesigner(policy, AcqMTV, init_sobol=0, init_center=False, acq_kwargs={"ttype": "mvar", "num_X_samples": default_num_X_samples}),
            "mtv_50": BTDesigner(
                policy, AcqMTV, init_sobol=0, init_center=False, acq_kwargs={"ttype": "mvar", "sample_type": "mh50", "num_X_samples": default_num_X_samples}
            ),
            "sr": BTDesigner(policy, qSimpleRegret),
            "ts": BTDesigner(policy, AcqTS, init_center=False),
            "ts_c": BTDesigner(policy, AcqTS, init_center=True),
            "ucb_c": BTDesigner(policy, qUpperConfidenceBound, acq_kwargs={"beta": 1}),
            "ucb": BTDesigner(policy, qUpperConfidenceBound, init_center=False, acq_kwargs={"beta": 1}),
            "ei_c": BTDesigner(policy, qNoisyExpectedImprovement, acq_kwargs={"X_baseline": None}),
            "ei": BTDesigner(policy, qNoisyExpectedImprovement, init_center=False, acq_kwargs={"X_baseline": None}),
            "mes": BTDesigner(policy, qMaxValueEntropy, acq_kwargs={"candidate_set": None}),
            "sobol_mes": BTDesigner(policy, qMaxValueEntropy, init_sobol=init_ax_default, acq_kwargs={"candidate_set": None}),
            "gibbon": BTDesigner(policy, qLowerBoundMaxValueEntropy, opt_sequential=True, acq_kwargs={"candidate_set": None}),
            "sobol_gibbon": BTDesigner(policy, qLowerBoundMaxValueEntropy, init_sobol=init_ax_default, acq_kwargs={"candidate_set": None}),
            "ax": AxDesigner(policy),
            "turbo": TuRBODesigner(policy, num_init=init_ax_default),
            "sobol_ei": BTDesigner(policy, qNoisyExpectedImprovement, init_sobol=init_ax_default, acq_kwargs={"X_baseline": None}),
            "sobol_ucb": BTDesigner(policy, qUpperConfidenceBound, init_sobol=init_ax_default, acq_kwargs={"beta": 1}),
        }

        self._add_ablations(policy, default_num_X_samples)

    def _add_ablations(self, policy, default_num_X_samples):
        self._designers = self._designers | {
            "mtv_eps=3.0": BTDesigner(
                policy, AcqMTV, init_sobol=0, init_center=False, acq_kwargs={"ttype": "msvar", "num_X_samples": default_num_X_samples, "beta": 0, "eps_0": 3.0}
            ),
            "mtv_eps=1.0": BTDesigner(
                policy, AcqMTV, init_sobol=0, init_center=False, acq_kwargs={"ttype": "msvar", "num_X_samples": default_num_X_samples, "beta": 0, "eps_0": 1.0}
            ),
            "mtv_eps=0.3": BTDesigner(
                policy, AcqMTV, init_sobol=0, init_center=False, acq_kwargs={"ttype": "msvar", "num_X_samples": default_num_X_samples, "beta": 0, "eps_0": 0.3}
            ),
            "mtv_eps=0.1": BTDesigner(
                policy, AcqMTV, init_sobol=0, init_center=False, acq_kwargs={"ttype": "msvar", "num_X_samples": default_num_X_samples, "beta": 0, "eps_0": 0.1}
            ),
            "mtv_eps=0.01": BTDesigner(
                policy, AcqMTV, init_sobol=0, init_center=False, acq_kwargs={"ttype": "msvar", "num_X_samples": default_num_X_samples, "beta": 0, "eps_0": 0.01}
            ),
            "mtv_no-opt": BTDesigner(
                policy,
                AcqMTV,
                init_sobol=0,
                init_center=False,
                acq_kwargs={"ttype": "ts", "num_X_samples": default_num_X_samples},
            ),
            "mtv_no-ic": BTDesigner(
                policy,
                AcqMTV,
                init_sobol=0,
                init_center=False,
                init_X_samples=False,
                acq_kwargs={"ttype": "msvar", "num_X_samples": default_num_X_samples, "beta": 0},
            ),
            "mtv_no-len-corr": BTDesigner(
                policy,
                AcqMTV,
                init_sobol=0,
                init_center=False,
                acq_kwargs={"ttype": "msvar", "num_X_samples": default_num_X_samples, "beta": 0, "lengthscale_correction": False},
            ),
            "mtv_no-pmax": BTDesigner(
                policy,
                AcqMTV,
                init_sobol=0,
                init_center=False,
                acq_kwargs={"ttype": "msvar", "sample_type": "sobol", "num_X_samples": default_num_X_samples, "beta": 0},
            ),
            "mtv_then_sr": [
                self._designers["mtv"],
                self._designers["sr"],
            ],
            "mtv_then_ucb": [
                self._designers["mtv"],
                self._designers["ucb"],
            ],
            "mtv_then_ei": [
                self._designers["mtv"],
                self._designers["ei"],
            ],
            "mtv_then_dpp": [
                self._designers["mtv"],
                self._designers["dpp"],
            ],
            "mtv_then_gibbon": [
                self._designers["mtv"],
                self._designers["gibbon"],
            ],
        }

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

        designers = self._designers[ttype]
        if not isinstance(designers, list):
            designers = [designers]
        if hasattr(designers[0], "init_center"):
            init_center = designers[0].init_center()
        else:
            init_center = True

        trace = []
        for _ in range(num_iterations):
            designer = designers[min(len(designers) - 1, self._i_iter)]

            best_in_batch = -1e99
            if init_center and self._i_iter == 0:
                if self._num_arms == 1:
                    data, d_time = self._iterate(self._center_designer, self._num_arms)
                else:
                    data_c, _ = self._iterate(self._center_designer, num_arms=1)
                    data, d_time = self._iterate(designer, self._num_arms - 1)
                    data.insert(0, data_c[0])
                    # data[np.random.choice(np.arange(self._num_arms))] = data_c[0]
            else:
                data, d_time = self._iterate(designer, self._num_arms)

            for datum in data:
                self._data.append(datum)
                best_in_batch = max(best_in_batch, datum.trajectory.rreturn)
                if self._datum_best is None or datum.trajectory.rreturn > self._datum_best.trajectory.rreturn:
                    self._datum_best = datum

            print(f"ITER: i_iter = {self._i_iter} d_time = {d_time:.2f} ret = {best_in_batch:.2f} ret_best = {self._datum_best.trajectory.rreturn:.2f}")
            sys.stdout.flush()
            trace.append(_TraceEntry(self._datum_best.trajectory.rreturn, d_time))
            self._i_iter += 1

        return trace
