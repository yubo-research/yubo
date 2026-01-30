import sys
import time
from dataclasses import dataclass

import numpy as np

from common.telemetry import Telemetry

from .datum import Datum
from .opt_trajectories import collect_denoised_trajectory, evaluate_for_best
from .trajectories import collect_trajectory

_INTERACTIVE_DEBUG = False
_SHOW_EVERY_N_ITER = 30


def _pareto_mask_max(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    assert y.ndim == 2, y.shape
    n = y.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        yi = y[i]
        ge = np.all(yi >= y, axis=1)
        gt = np.any(yi > y, axis=1)
        dom = ge & gt
        dom[i] = False
        keep[dom] = False
    return keep


def _pareto_mask_min(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    assert y.ndim == 2, y.shape
    n = y.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        yi = y[i]
        le = np.all(yi <= y, axis=1)
        lt = np.any(yi < y, axis=1)
        dom = le & lt
        dom[i] = False
        keep[dom] = False
    return keep


@dataclass
class _TraceEntry:
    rreturn: float
    rreturn_decision: float
    dt_prop: float
    dt_eval: float


class Optimizer:
    def __init__(
        self,
        collector,
        *,
        env_conf,
        policy,
        num_arms,
        num_denoise_measurement=None,
        num_denoise_passive=None,
    ):
        self._collector = collector
        self._env_conf = env_conf
        self.best_policy = policy
        self._num_arms = num_arms
        self._num_denoise = num_denoise_measurement
        self._num_denoise_passive_eval = num_denoise_passive
        self.num_params = policy.num_params()
        self.r_best_est = -1e99
        self.y_best = None
        self.best_datum = None

        self._data = []
        self._i_iter = 0
        self._i_noise = 0
        self._cum_dt_proposing = 0

        self._collector(
            f"PROBLEM: env = {env_conf.env_name} num_params = {policy.num_params()}"
        )

        self._ret_viz = -1e99
        self._telemetry = Telemetry()
        self._ref_point = None

    def _iterate(self, designer, num_arms):
        self._telemetry.reset()
        t_0 = time.time()
        policies = designer(self._data, num_arms, telemetry=self._telemetry)
        t_f = time.time()
        dt_prop = t_f - t_0

        data = []
        t_0 = time.time()
        for policy in policies:
            if self._env_conf.frozen_noise:
                i_noise = None
            else:
                i_noise = self._i_noise
                if self._num_denoise is None:
                    delta = 1
                else:
                    delta = self._num_denoise
                self._i_noise += delta
            traj, noise_seed = collect_denoised_trajectory(
                self._env_conf, policy, self._num_denoise, i_noise
            )
            if _INTERACTIVE_DEBUG and self._num_denoise == 1:
                r = np.asarray(traj.rreturn)
                if r.ndim == 0 and float(r) > self._ret_viz:
                    self._policy_viz = policy.clone()
                    self._ret_viz = float(r)
                    self._noise_seed_viz = noise_seed
            data.append(Datum(designer, policy, None, traj))
        tf = time.time()
        dt_eval = tf - t_0

        return data, dt_prop, dt_eval

    def collect_trace(
        self, designer_name, max_iterations, max_proposal_seconds=np.inf, deadline=None
    ):
        self.initialize(designer_name)
        num_iterations = 0
        while (
            num_iterations < max_iterations
            and self._cum_dt_proposing < max_proposal_seconds
        ):
            if deadline is not None and time.time() >= deadline:
                break
            self.iterate()
            num_iterations += 1
        self.stop()
        return self._trace

    def initialize(self, designer_name):
        from .designers import Designers

        designers = Designers(self.best_policy, self._num_arms)
        self._opt_designers = designers.create(designer_name)

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
        ret_batch = np.asarray(ret_batch)

        if _INTERACTIVE_DEBUG:
            if ret_batch.ndim == 1 and self._i_iter % _SHOW_EVERY_N_ITER == 0:
                print("VIZ:", self._ret_viz, self.r_best_est, ret_batch.max())
                collect_trajectory(
                    self._env_conf,
                    self._policy_viz,
                    noise_seed=self._noise_seed_viz,
                    show_frames=True,
                )

        if ret_batch.ndim <= 1:
            did_update_best = False
            used_designer_best = False
            if hasattr(designer, "best_datum") and callable(
                getattr(designer, "best_datum")
            ):
                datum_best = designer.best_datum()
                if datum_best is not None:
                    decision_best = float(datum_best.trajectory.get_decision_rreturn())
                    if decision_best > float(self.r_best_est):
                        self.r_best_est = decision_best
                        self.best_datum = datum_best
                        self.best_policy = self.best_datum.policy.clone()
                        did_update_best = True
                    self.r_best_est = max(self.r_best_est, decision_best)
                    used_designer_best = True
            if not used_designer_best:
                decision_batch = []
                for datum in data:
                    decision_batch.append(datum.trajectory.get_decision_rreturn())
                decision_batch = np.asarray(decision_batch, dtype=np.float64)
                best_idx = int(np.argmax(decision_batch))
                decision_best_batch = float(decision_batch[best_idx])

                if float(decision_best_batch) > float(self.r_best_est):
                    self.r_best_est = float(decision_best_batch)
                    self.best_datum = data[best_idx]
                    self.best_policy = self.best_datum.policy.clone()
                    did_update_best = True

                self.r_best_est = max(self.r_best_est, float(decision_best_batch))

            if self._num_denoise_passive_eval is None:
                # Frozen noise: track best raw rreturn seen so far (monotonically increasing)
                best_raw_rreturn_batch = float(np.max(ret_batch))
                if self.y_best is None:
                    self.y_best = best_raw_rreturn_batch
                else:
                    self.y_best = max(float(self.y_best), best_raw_rreturn_batch)
            else:
                if did_update_best or self.y_best is None:
                    self.y_best = float(
                        evaluate_for_best(
                            self._env_conf,
                            self.best_policy,
                            self._num_denoise_passive_eval,
                        )
                    )

            ret_eval = float(self.y_best)

            y_best_s = f"{float(self.y_best):.3f}"
            ret_best_s = f"{float(self.r_best_est):.3f}"
            ret_eval_s = f"{float(ret_eval):.3f}"
        else:
            assert ret_batch.ndim == 2, ret_batch.shape
            num_metrics = int(ret_batch.shape[1])
            assert num_metrics >= 2, ret_batch.shape
            assert self._num_denoise_passive_eval is None, (
                "NYI: passive denoise for vector returns"
            )

            if num_metrics == 2:
                if self._ref_point is None:
                    from analysis.ref_point import SobolRefPoint

                    noise_seed_0 = (
                        0
                        if self._env_conf.noise_seed_0 is None
                        else int(self._env_conf.noise_seed_0)
                    )
                    seed = int(self._env_conf.problem_seed) + 99991
                    self._ref_point = SobolRefPoint(
                        num_cal=max(128, 10 * int(self._num_arms)),
                        seed=seed,
                        num_denoise=self._num_denoise,
                        noise_seed_0=noise_seed_0,
                        std_margin_scale=0.1,
                    ).compute(
                        self._env_conf,
                        policy=self.best_policy.clone()
                        if self.best_policy is not None
                        else None,
                    )
                    self._collector(
                        f"REF_POINT: ref = {np.array2string(self._ref_point, precision=6, floatmode='fixed')}"
                    )

                all_y = np.asarray(
                    [
                        np.asarray(d.trajectory.rreturn, dtype=np.float64)
                        for d in self._data
                    ],
                    dtype=np.float64,
                )
                import torch
                from botorch.utils.multi_objective.hypervolume import Hypervolume
                from botorch.utils.multi_objective.pareto import is_non_dominated

                y_t = torch.as_tensor(all_y, dtype=torch.double)
                nd_mask = is_non_dominated(y_t)
                front = y_t[nd_mask]
                if front.numel() == 0:
                    hv = 0.0
                    self.y_best = np.full(num_metrics, np.nan)
                    self.best_datum = None
                    self.best_policy = None
                else:
                    ref_t = torch.as_tensor(self._ref_point, dtype=torch.double)
                    hv = float(Hypervolume(ref_t).compute(front))
                    front_np = front.cpu().numpy()
                    normalized = (front_np - self._ref_point) / np.abs(
                        self._ref_point + 1e-9
                    )
                    scores = normalized.sum(axis=1)
                    best_idx = int(np.argmax(scores))
                    self.y_best = front_np[best_idx]
                    nd_idx = torch.nonzero(nd_mask, as_tuple=False).reshape(-1)
                    best_data_idx = int(nd_idx[best_idx].item())
                    self.best_datum = self._data[best_data_idx]
                    self.best_policy = self.best_datum.policy.clone()

                self.r_best_est = max(self.r_best_est, float(hv))
                ret_eval = float(hv)
            else:
                ret_0 = np.asarray(ret_batch[:, 0], dtype=np.float64)
                best_idx = int(np.argmax(ret_0))
                ret_best_batch = float(ret_0[best_idx])
                if float(ret_best_batch) > float(self.r_best_est):
                    self.r_best_est = float(ret_best_batch)
                    self.y_best = np.asarray(ret_batch[best_idx], dtype=np.float64)
                    self.best_datum = data[best_idx]
                    self.best_policy = self.best_datum.policy.clone()
                ret_eval = float(self.r_best_est)

            y_best_s = np.array2string(self.y_best, precision=3, floatmode="fixed")
            ret_best_s = f"{float(self.r_best_est):.6f}"
            ret_eval_s = f"{float(ret_eval):.6f}"

        cum_time = time.time() - self._t_0
        self._cum_dt_proposing += dt_prop
        if ret_eval > -1e98:
            self._collector(
                f"ITER: i_iter = {self._i_iter} cum_time = {cum_time:.2f} dt_eval = {dt_eval:.3f} dt_prop = {dt_prop:.3f} {self._telemetry.format()} cum_dt_prop = {self._cum_dt_proposing:.3f} y_best = {y_best_s} ret_best = {ret_best_s} ret_eval = {ret_eval_s}"
            )
        sys.stdout.flush()
        self._trace.append(
            _TraceEntry(float(ret_eval), float(self.r_best_est), dt_prop, dt_eval)
        )
        self._i_iter += 1
        self.last_designer = designer
        return self._trace

    def stop(self):
        for designer in self._opt_designers:
            if hasattr(designer, "stop"):
                designer.stop()
