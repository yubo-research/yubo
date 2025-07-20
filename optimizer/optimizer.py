import sys
import time
from dataclasses import dataclass

import numpy as np

from .center_designer import CenterDesigner
from .datum import Datum
from .designers import Designers
from .trajectories import Trajectory, collect_trajectory

_INTERACTIVE_DEBUG = True
_SHOW_EVERY_N_ITER = 30


@dataclass
class _TraceEntry:
    rreturn: float
    dt_prop: float
    dt_eval: float


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

    def _collect_trajectory(self, policy, i_noise=None, denoise_seed=0):
        if i_noise is None:
            noise_seed = 0
        else:
            noise_seed = i_noise

        noise_seed += self._env_conf.noise_seed_0 + denoise_seed
        self._last_noise_seed = noise_seed

        # policy_a = policy.clone()
        # policy_b = policy.clone()
        # a = collect_trajectory(self._env_conf, policy_a, noise_seed=noise_seed)
        # b = collect_trajectory(self._env_conf, policy_b, noise_seed=noise_seed)
        # assert a.rreturn == b.rreturn, f"{a.rreturn} != {b.rreturn}"
        # return a
        return collect_trajectory(self._env_conf, policy, noise_seed=noise_seed)

    def _collect_denoised_trajectory(self, policy, i_noise=None):
        if self._num_denoise is not None:
            if self._num_denoise == 1:
                # TODO: Think about what to do with states and actions when num_denoise > 1)
                policy_orig = policy.clone()
                traj = self._collect_trajectory(policy, denoise_seed=0)
                if _INTERACTIVE_DEBUG:
                    if traj.rreturn > self.r_best_est:
                        self._policy_viz = policy_orig.clone()
                        self._ret_viz = traj.rreturn
                        self._noise_seed_viz = self._last_noise_seed

                return traj
            rreturn = self._mean_return_over_runs(policy)
            return Trajectory(rreturn, None, None)

        return self._collect_trajectory(policy, i_noise=i_noise)

    def _iterate(self, designer, num_arms):
        t0 = time.time()
        policies = designer(self._data, num_arms)
        tf = time.time()
        dt_prop = tf - t0

        data = []
        X = []

        t_0 = time.time()
        for policy in policies:
            if self._env_conf.frozen_noise:
                i_noise = None
            else:
                i_noise = self._i_noise
                self._i_noise += 1
            traj = self._collect_denoised_trajectory(policy, i_noise)
            data.append(Datum(designer, policy, None, traj))
            X.append(policy.get_params())
        tf = time.time()
        dt_eval = tf - t_0

        return data, dt_prop, dt_eval

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

    def collect_trace(self, designer_name, max_iterations, max_proposal_seconds=np.inf):
        self.initialize(designer_name)
        num_iterations = 0
        while num_iterations < max_iterations and self._cum_dt_proposing < max_proposal_seconds:
            self.iterate()
            num_iterations += 1
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

        data, dt_prop, dt_eval = self._iterate(designer, self._num_arms)

        ret_batch = []
        for datum in data:
            self._data.append(datum)
            ret_batch.append(datum.trajectory.rreturn)

        # one ret for each *arm* (nothing to do with num_denoise)
        ret_batch = np.array(ret_batch)

        if _INTERACTIVE_DEBUG:
            # if ret_batch.max() > self.r_best_est:
            if self._i_iter % _SHOW_EVERY_N_ITER == 0:
                print("RET:", self._ret_viz, self.r_best_est, ret_batch.max())
                collect_trajectory(self._env_conf, self._policy_viz, noise_seed=self._noise_seed_viz, show_frames=True)

        self.r_best_est = max(self.r_best_est, ret_batch.max())
        if self._b_cumulative_reward:
            self._r_cumulative += ret_batch.mean()
            ret_eval = self._r_cumulative / (1 + self._i_iter)
        else:
            ret_eval = self.r_best_est

        cum_time = time.time() - self._t_0
        self._cum_dt_proposing += dt_prop
        self._collector(
            f"ITER: i_iter = {self._i_iter} cum_time = {cum_time:.2f} dt_eval = {dt_eval:.3f} dt_prop = {dt_prop:.3f} cum_dt_prop = {self._cum_dt_proposing:.3f} ret_max = {ret_batch.max():.3f} ret_mean = {ret_batch.mean():.3f} ret_best = {self.r_best_est:.3f} ret_eval = {ret_eval:.3f}"
        )
        sys.stdout.flush()
        self._trace.append(_TraceEntry(ret_eval, dt_prop, dt_eval))
        self._i_iter += 1
        self.last_designer = designer
        return self._trace

    def stop(self):
        for designer in self._opt_designers:
            if hasattr(designer, "stop"):
                designer.stop()
