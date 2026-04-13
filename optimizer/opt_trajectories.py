from typing import NamedTuple

import numpy as np

from .trajectories import collect_trajectory
from .trajectory import Trajectory


class _MeanReturnResult(NamedTuple):
    mean: float
    se: float
    all_same: bool
    num_steps_total: int


def collect_trajectory_with_noise(env_conf, policy, i_noise=None, denoise_seed=0):
    if i_noise is None:
        noise_seed = 0
    else:
        noise_seed = i_noise
    noise_seed += env_conf.noise_seed_0 + denoise_seed

    traj = collect_trajectory(env_conf, policy, noise_seed=noise_seed)
    return traj, noise_seed


def mean_return_over_runs(env_conf, policy, num_denoise, i_noise=None):
    rets = []
    num_steps_total = 0
    for i in range(num_denoise):
        traj, _ = collect_trajectory_with_noise(env_conf, policy, i_noise=i_noise, denoise_seed=i)
        rets.append(traj.rreturn)
        num_steps_total += int(getattr(traj, "num_steps", 0))
    std_rets = np.std(rets)
    all_same = len(rets) > 1 and std_rets == 0
    return _MeanReturnResult(
        mean=float(np.mean(rets)),
        se=float(std_rets / np.sqrt(len(rets))),
        all_same=bool(all_same),
        num_steps_total=int(num_steps_total),
    )


def collect_denoised_trajectory(env_conf, policy, num_denoise, i_noise=None):
    if num_denoise is None:
        return collect_trajectory_with_noise(env_conf, policy, i_noise=i_noise)

    if num_denoise == 1:
        return collect_trajectory_with_noise(env_conf, policy, i_noise=i_noise, denoise_seed=0)

    mean_result = mean_return_over_runs(env_conf, policy, num_denoise, i_noise=i_noise)
    rreturn = mean_result.mean
    rreturn_se = mean_result.se

    if env_conf.frozen_noise:
        rreturn_se = None

    return Trajectory(rreturn, None, None, rreturn_se, num_steps=int(mean_result.num_steps_total)), None


def evaluate_for_best(env_conf, policy, num_denoise_passiveuation, *, i_noise=99999, return_steps=False):
    mean_result = mean_return_over_runs(env_conf, policy, num_denoise_passiveuation, i_noise=i_noise)
    if return_steps:
        return float(mean_result.mean), int(mean_result.num_steps_total)
    return float(mean_result.mean)
