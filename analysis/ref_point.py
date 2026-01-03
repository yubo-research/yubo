from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from optimizer.trajectories import collect_trajectory
from problems.env_conf import default_policy


def _as_2d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 0:
        return y.reshape(1, 1)
    if y.ndim == 1:
        return y.reshape(-1, 1)
    if y.ndim == 2:
        return y
    assert False, y.shape


def _eval_mean_return(env_conf, policy, *, num_denoise: Optional[int], noise_seed_0: int, i_noise: int) -> np.ndarray:
    if num_denoise is None:
        num_denoise = 1
    assert num_denoise >= 1, num_denoise

    rets = []
    for i in range(num_denoise):
        traj = collect_trajectory(env_conf, policy, noise_seed=noise_seed_0 + i_noise + i)
        rets.append(traj.rreturn)
    y = np.mean(np.asarray(rets), axis=0)
    return np.asarray(y)


@dataclass(frozen=True)
class SobolRefPoint:
    num_cal: int
    seed: int
    num_denoise: Optional[int] = None
    noise_seed_0: int = 0
    std_margin_scale: float = 0.1
    directions: Optional[np.ndarray] = None

    def compute(self, env_conf, *, policy=None) -> np.ndarray:
        assert self.num_cal >= 1, self.num_cal
        assert np.isfinite(self.std_margin_scale), self.std_margin_scale
        assert self.std_margin_scale >= 0.0, self.std_margin_scale

        p = policy if policy is not None else default_policy(env_conf)
        x_0 = np.asarray(p.get_params()).copy()

        d = int(p.num_params())
        sobol = torch.quasirandom.SobolEngine(d, scramble=True, seed=int(self.seed))
        u = sobol.draw(self.num_cal).cpu().numpy()
        x = 2.0 * u - 1.0

        ys = []
        stride = 1 if self.num_denoise in (None, 1) else int(self.num_denoise)
        for i in range(self.num_cal):
            p.set_params(x[i])
            i_noise = 0 if getattr(env_conf, "frozen_noise", True) else int(i) * stride
            ys.append(_eval_mean_return(env_conf, p, num_denoise=self.num_denoise, noise_seed_0=int(self.noise_seed_0), i_noise=i_noise))

        p.set_params(x_0)

        y = _as_2d(np.asarray(ys))

        if self.directions is None:
            y_adj = y
        else:
            directions = np.asarray(self.directions, dtype=np.float64).reshape(1, -1)
            assert directions.shape[1] == y.shape[1], (directions.shape, y.shape)
            y_adj = y * directions

        y_min = np.min(y_adj, axis=0)
        y_std = np.std(y_adj, axis=0)
        ref = y_min - self.std_margin_scale * y_std
        assert np.all(np.isfinite(ref)), ref
        return ref
