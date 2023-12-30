import sys
import time
from dataclasses import dataclass

import numpy as np
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
)
from botorch.acquisition.monte_carlo import (
    qNoisyExpectedImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)

from bo.acq_dpp import AcqDPP
from bo.acq_min_dist import AcqMinDist
from bo.acq_mtv import AcqMTV
from bo.acq_var import AcqVar

from .ax_designer import AxDesigner
from .bt_designer import BTDesigner
from .center_designer import CenterDesigner
from .cma_designer import CMADesigner
from .datum import Datum
from .random_designer import RandomDesigner
from .sobol_designer import SobolDesigner
from .trajectories import Trajectory, collect_trajectory
from .turbo_designer import TuRBODesigner


@dataclass
class _TraceEntry:
    rreturn: float
    time_iteration_seconds: float


class Optimizer:
    def __init__(self, env_conf, policy, num_arms, arm_selector, num_obs=1, num_denoise=None, cb_trace=None):
        self._env_conf = env_conf
        self._num_arms = num_arms
        self._num_obs = num_obs
        self._num_denoise = num_denoise
        self._arm_selector = arm_selector
        self.num_params = policy.num_params()

        self._data = []
        self._i_iter = 0
        self._i_noise = 0
        self._center_designer = CenterDesigner(policy)

        print(f"PROBLEM: env = {env_conf.env_name} num_params = {policy.num_params()}")
        init_ax_default = max(5, 2 * policy.num_params())
        default_num_X_samples = max(64, 10 * self._num_arms)
        # default_num_Y_samples = 512

        self._designers = {
            # Optimization packages
            "cma": CMADesigner(policy),
            "ax": AxDesigner(policy),
            # 1D only
            "maximin": BTDesigner(policy, lambda m: AcqMinDist(m, toroidal=False), init_center=False),
            "maximin-toroidal": BTDesigner(policy, lambda m: AcqMinDist(m, toroidal=True), init_center=False),
            "variance": BTDesigner(policy, AcqVar, init_center=False),
            # Init only, no surrogate, all exploration
            "random": RandomDesigner(policy, init_center=False),
            "random_c": RandomDesigner(policy, init_center=True),
            "sobol": SobolDesigner(policy, init_center=False),
            "sobol_c": SobolDesigner(policy, init_center=True),
            # All exploitation
            "sr": BTDesigner(policy, qSimpleRegret, init_center=False),
            # Various methods, first batch is Sobol
            "ucb": BTDesigner(policy, qUpperConfidenceBound, init_center=False, acq_kwargs={"beta": 1}),
            "ei": BTDesigner(policy, qNoisyExpectedImprovement, init_center=False, acq_kwargs={"X_baseline": None}),
            "gibbon": BTDesigner(policy, qLowerBoundMaxValueEntropy, init_center=False, opt_sequential=True, acq_kwargs={"candidate_set": None}),
            "turbo": TuRBODesigner(policy, num_init=init_ax_default),
            "dpp": BTDesigner(policy, AcqDPP, init_sobol=1, init_center=False, acq_kwargs={"num_X_samples": default_num_X_samples}),
            # Force a center point into the initialization
            "ucb_c": BTDesigner(policy, qUpperConfidenceBound, init_center=True, acq_kwargs={"beta": 1}),
            "ei_c": BTDesigner(policy, qNoisyExpectedImprovement, init_center=True, acq_kwargs={"X_baseline": None}),
            "gibbon_c": BTDesigner(policy, qLowerBoundMaxValueEntropy, init_center=True, opt_sequential=True, acq_kwargs={"candidate_set": None}),
            "dpp_c": BTDesigner(policy, AcqDPP, init_sobol=1, init_center=True, acq_kwargs={"num_X_samples": default_num_X_samples}),
            # MTV
            "mtv": BTDesigner(policy, AcqMTV, init_sobol=0, init_center=False, acq_kwargs={"ttype": "mvar", "num_X_samples": default_num_X_samples}),
            # Long sobol init, sequential opt
            "sobol_ucb": BTDesigner(policy, qUpperConfidenceBound, init_center=False, init_sobol=init_ax_default, acq_kwargs={"beta": 1}),
            "sobol_ei": BTDesigner(policy, qNoisyExpectedImprovement, init_center=False, init_sobol=init_ax_default, acq_kwargs={"X_baseline": None}),
            "sobol_gibbon": BTDesigner(policy, qLowerBoundMaxValueEntropy, init_center=False, init_sobol=init_ax_default, acq_kwargs={"candidate_set": None}),
        }

        self._add_ablations(policy, default_num_X_samples)

    def _add_ablations(self, policy, default_num_X_samples):
        self._designers = self._designers | {
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
                acq_kwargs={"ttype": "mvar", "num_X_samples": default_num_X_samples},
            ),
            "mtv_no-pstar": BTDesigner(
                policy,
                AcqMTV,
                init_sobol=0,
                init_center=False,
                acq_kwargs={"ttype": "mvar", "sample_type": "sobol", "num_X_samples": default_num_X_samples},
            ),
            # MTV for first batch only
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
        # Use a different noise seed every time we collect a trajetory.
        noise_seed = self._env_conf.noise_seed_0 + self._i_noise
        self._i_noise += 1
        traj = collect_trajectory(self._env_conf, policy, noise_seed=noise_seed)
        return traj

    def _iterate(self, designer, num_arms):
        t0 = time.time()
        policies = designer(self._data, num_arms)
        tf = time.time()
        data = []
        X = []
        for policy in policies:
            if self._num_obs == 1:
                traj = self.collect_trajectory(policy)
                data.append(Datum(designer, policy, None, traj))
            else:
                rreturns = []
                for _ in range(self._num_obs):
                    traj = self.collect_trajectory(policy)
                    rreturns.append(traj.rreturn)
                data.append(Datum(designer, policy, None, Trajectory(np.mean(rreturns), "n/a when num_obs>1", "n/a when num_obs>1")))
            X.append(policy.get_params())

        return data, tf - t0

    def _denoise(self, policy):
        rets = []
        for i in range(self._num_denoise):
            traj = self.collect_trajectory(policy)
            rets.append(traj.rreturn)
        if np.std(rets) == 0:
            print(f"WARNING: All rets are the same {rets}")
            # assert np.std(rets) > 0, rets
        return np.mean(rets)

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

            if init_center and self._i_iter == 0:
                assert False
                if self._num_arms == 1:
                    data, d_time = self._iterate(self._center_designer, self._num_arms)
                else:
                    data_c, _ = self._iterate(self._center_designer, num_arms=1)
                    data, d_time = self._iterate(designer, self._num_arms - 1)
                    data.insert(0, data_c[0])
                    # data[np.random.choice(np.arange(self._num_arms))] = data_c[0]
            else:
                data, d_time = self._iterate(designer, self._num_arms)

            ret_batch = []
            for datum in data:
                self._data.append(datum)
                ret_batch.append(datum.trajectory.rreturn)

            policy_best, r_best_est = self._arm_selector(self._data)
            if self._num_denoise is None:
                ret_eval = r_best_est
            else:
                ret_eval = self._denoise(policy_best)

            ret_batch = np.array(ret_batch)

            print(
                f"ITER: i_iter = {self._i_iter} d_time = {d_time:.2f} ret_max = {ret_batch.max():.2f} ret_mean = {ret_batch.mean():.2f} ret_best = {r_best_est:.2f} ret_eval = {ret_eval:.2f}"
            )
            sys.stdout.flush()
            trace.append(_TraceEntry(ret_eval, d_time))
            self._i_iter += 1

        return trace
