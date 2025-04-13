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
    def __init__(self, collector, *, env_conf, policy, num_arms, num_denoise_measurement=None):
        self._collector = collector
        self._env_conf = env_conf
        self._num_arms = num_arms
        self._num_denoise = num_denoise_measurement
        self.num_params = policy.num_params()
        self.r_best_est = -1e99
        self._r_cumulative = 0
        self._b_cumulative_reward = False

        self._data = []
        self._i_iter = 0
        self._i_noise = 0
        self._cum_dt_proposing = 0
        self._center_designer = CenterDesigner(policy)

        self._collector(f"PROBLEM: env = {env_conf.env_name} num_params = {policy.num_params()}")
        self._designers = Designers(policy, num_arms)

    def _collect_trajectory(self, policy, denoise_seed=0):
        if self._env_conf.frozen_noise:
            noise_seed = 0
        else:
            noise_seed = self._i_noise
            self._i_noise += 1
        noise_seed += self._env_conf.noise_seed_0 + denoise_seed

        return collect_trajectory(self._env_conf, policy, noise_seed=noise_seed)

    def _collect_denoised_trajectory(self, policy):
        if self._num_denoise is not None:
            rreturn = self._mean_return_over_runs(policy)
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

    def _mean_return_over_runs(self, policy):
        rets = []
        for i in range(self._num_denoise):
            # We follow the denoising logic in https://papers.nips.cc/paper_files/paper/2019/file/6c990b7aca7bc7058f5e98ea909e924b-Paper.pdf
            #  and use a *fixed* set of seeds for every evaluation.
            # That tends to make a problem much easier. We/you should also study problems
            #  where the noise is not fixed.
            traj = self._collect_trajectory(policy, denoise_seed=i)
            rets.append(traj.rreturn)
        if len(rets) > 1 and np.std(rets) == 0:
            self._collector(f"WARNING: All rets are the same {rets}")
            # assert np.std(rets) > 0, rets
        return np.mean(rets)

    def collect_trace(self, designer_name, num_iterations):
        # from pympler import tracker

        # tr = tracker.SummaryTracker()

        self.initialize(designer_name)
        for _ in range(num_iterations):
            self.iterate()
        self.stop()
        return self._trace

    def initialize(self, designer_name):
        self._opt_designers = self._designers.create(designer_name)

        if not isinstance(self._opt_designers, list):
            self._opt_designers = [self._opt_designers]

        self._trace = []
        self._t_0 = time.time()

    def iterate(self):
        designer = self._opt_designers[min(len(self._opt_designers) - 1, self._i_iter)]

        data, dt_proposing = self._iterate(designer, self._num_arms)

        ret_batch = []
        for datum in data:
            self._data.append(datum)
            ret_batch.append(datum.trajectory.rreturn)

        # one ret for each *arm* (nothing to do with num_denoise)
        ret_batch = np.array(ret_batch)

        self.r_best_est = max(self.r_best_est, ret_batch.max())
        if self._b_cumulative_reward:
            self._r_cumulative += ret_batch.mean()
            ret_eval = self._r_cumulative / (1 + self._i_iter)
        else:
            ret_eval = self.r_best_est

        cum_time = time.time() - self._t_0
        self._cum_dt_proposing += dt_proposing
        self._collector(
            f"ITER: i_iter = {self._i_iter} cum_time = {cum_time:.2f} dt_prop = {dt_proposing:.3f} cum_dt_prop = {self._cum_dt_proposing:.3f} ret_max = {ret_batch.max():.3f} ret_mean = {ret_batch.mean():.3f} ret_best = {self.r_best_est:.3f} ret_eval = {ret_eval:.3f}"
        )
        sys.stdout.flush()
        self._trace.append(_TraceEntry(ret_eval, dt_proposing))
        self._i_iter += 1
        self.last_designer = designer
        return self._trace

    def stop(self):
        for designer in self._opt_designers:
            if hasattr(designer, "stop"):
                designer.stop()
