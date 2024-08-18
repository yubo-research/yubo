import sys
import time
from dataclasses import dataclass

import numpy as np

from .center_designer import CenterDesigner
from .datum import Datum
from .designers import Designers
from .trajectories import Trajectory, collect_trajectory


@dataclass
class _TraceEntry:
    rreturn: float
    time_iteration_seconds: float


class Optimizer:
    def __init__(self, collector, *, env_conf, policy, num_arms, arm_selector, num_denoise=None):
        self._collector = collector
        self._env_conf = env_conf
        self._num_arms = num_arms
        self._num_denoise = num_denoise
        self._arm_selector = arm_selector
        self.num_params = policy.num_params()

        self._data = []
        self._i_iter = 0
        self._i_noise = 0
        self._center_designer = CenterDesigner(policy)

        self._collector(f"PROBLEM: env = {env_conf.env_name} num_params = {policy.num_params()}")
        self._designers = Designers(policy, num_arms)

    def _collect_trajectory(self, policy, denoise_seed=0):
        # Use a different noise seed every time we collect a trajetory.
        noise_seed = self._env_conf.noise_seed_0 + self._i_noise + denoise_seed
        return collect_trajectory(self._env_conf, policy, noise_seed=noise_seed)

    def _collect_denoised_trajectory(self, policy):
        if self._num_denoise is not None:
            rreturn = self._denoise(policy)
            return Trajectory(rreturn, None, None)
        return self._collect_trajectory(policy)

    def _iterate(self, designer, num_arms):
        t0 = time.time()
        policies = designer(self._data, num_arms)
        tf = time.time()
        data = []
        X = []
        for policy in policies:
            traj = self._collect_denoised_trajectory(policy)
            data.append(Datum(designer, policy, None, traj))
            X.append(policy.get_params())

        return data, tf - t0

    def _denoise(self, policy):
        rets = []
        for i in range(self._num_denoise):
            traj = self._collect_trajectory(policy, denoise_seed=i)
            rets.append(traj.rreturn)
        if np.std(rets) == 0:
            self._collector(f"WARNING: All rets are the same {rets}")
            # assert np.std(rets) > 0, rets
        return np.mean(rets)

    def collect_trace(self, designer_name, num_iterations):
        designers = self._designers.create(designer_name)
        if not isinstance(designers, list):
            designers = [designers]

        trace = []
        t_0 = time.time()
        for _ in range(num_iterations):
            self._i_noise += 1
            designer = designers[min(len(designers) - 1, self._i_iter)]

            data, d_time = self._iterate(designer, self._num_arms)

            ret_batch = []
            for datum in data:
                self._data.append(datum)
                ret_batch.append(datum.trajectory.rreturn)

            policy_best, self.r_best_est = self._arm_selector(self._data)
            ret_eval = self.r_best_est
            ret_batch = np.array(ret_batch)

            cum_time = time.time() - t_0
            self._collector(
                f"ITER: i_iter = {self._i_iter} cum_time = {cum_time:.2f} d_time = {d_time:.2f} ret_max = {ret_batch.max():.3f} ret_mean = {ret_batch.mean():.3f} ret_best = {self.r_best_est:.3f} ret_eval = {ret_eval:.3f}"
            )
            sys.stdout.flush()
            trace.append(_TraceEntry(ret_eval, d_time))
            self._i_iter += 1
            self.last_designer = designer

        for designer in designers:
            if hasattr(designer, "stop"):
                designer.stop()
        return trace
