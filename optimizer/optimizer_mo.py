"""Multi-objective return handling for :class:`~optimizer.optimizer.Optimizer`."""

import numpy as np


class OptimizerMultiObjectiveMixin:
    def _handle_multi_objective_returns(self, data, ret_batch, num_metrics):
        if num_metrics == 2:
            ret_eval = self._handle_hypervolume(num_metrics)
        else:
            ret_eval = self._handle_first_objective(data, ret_batch)
        from .optimizer_types import ReturnSummary

        return ReturnSummary(
            ret_eval=float(ret_eval),
            y_best_s=np.array2string(self.y_best, precision=3, floatmode="fixed"),
            ret_best_s=f"{float(self.r_best_est):.6f}",
            ret_eval_s=f"{float(ret_eval):.6f}",
        )

    def _handle_hypervolume(self, num_metrics):
        if self._ref_point is None:
            self._init_ref_point()
        import torch
        from botorch.utils.multi_objective.hypervolume import Hypervolume
        from botorch.utils.multi_objective.pareto import is_non_dominated

        all_y = np.asarray(
            [np.asarray(d.trajectory.rreturn, dtype=np.float64) for d in self._data],
            dtype=np.float64,
        )
        y_t = torch.as_tensor(all_y, dtype=torch.double)
        nd_mask = is_non_dominated(y_t)
        front = y_t[nd_mask]
        if front.numel() == 0:
            hv = 0.0
            self.y_best = np.full(num_metrics, np.nan)
            self.best_datum = self.best_policy = None
        else:
            ref_t = torch.as_tensor(self._ref_point, dtype=torch.double)
            hv = float(Hypervolume(ref_t).compute(front))
            front_np = front.cpu().numpy()
            scores = ((front_np - self._ref_point) / np.abs(self._ref_point + 1e-9)).sum(axis=1)
            best_idx = int(np.argmax(scores))
            self.y_best = front_np[best_idx]
            nd_idx = torch.nonzero(nd_mask, as_tuple=False).reshape(-1)
            self.best_datum = self._data[int(nd_idx[best_idx].item())]
            self.best_policy = self.best_datum.policy.clone()
        self.r_best_est = max(self.r_best_est, float(hv))
        return float(hv)

    def _init_ref_point(self):
        from analysis.ref_point import SobolRefPoint

        noise_seed_0 = 0 if self._env_conf.noise_seed_0 is None else int(self._env_conf.noise_seed_0)
        seed = int(self._env_conf.problem_seed) + 99991
        self._ref_point = SobolRefPoint(
            num_cal=max(128, 10 * int(self._num_arms)),
            seed=seed,
            num_denoise=self._num_denoise,
            noise_seed_0=noise_seed_0,
            std_margin_scale=0.1,
        ).compute(
            self._env_conf,
            policy=self.best_policy.clone() if self.best_policy is not None else None,
        )
        self._collector(f"REF_POINT: ref = {np.array2string(self._ref_point, precision=6, floatmode='fixed')}")

    def _handle_first_objective(self, data, ret_batch):
        ret_0 = np.asarray(ret_batch[:, 0], dtype=np.float64)
        best_idx = int(np.argmax(ret_0))
        ret_best_batch = float(ret_0[best_idx])
        if ret_best_batch > float(self.r_best_est):
            self.r_best_est = ret_best_batch
            self.y_best = np.asarray(ret_batch[best_idx], dtype=np.float64)
            self.best_datum = data[best_idx]
            self.best_policy = self.best_datum.policy.clone()
        return float(self.r_best_est)
