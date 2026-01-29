import numpy as np

from .trajectories import Trajectory, collect_trajectory


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
    for i in range(num_denoise):
        traj, _ = collect_trajectory_with_noise(
            env_conf, policy, i_noise=i_noise, denoise_seed=i
        )
        rets.append(traj.rreturn)
    std_rets = np.std(rets)
    all_same = len(rets) > 1 and std_rets == 0
    return np.mean(rets), std_rets / np.sqrt(len(rets)), all_same


def collect_denoised_trajectory(env_conf, policy, num_denoise, i_noise=None):
    if num_denoise is None:
        return collect_trajectory_with_noise(env_conf, policy, i_noise=i_noise)

    if num_denoise == 1:
        return collect_trajectory_with_noise(
            env_conf, policy, i_noise=i_noise, denoise_seed=0
        )

    rreturn, rreturn_se, _ = mean_return_over_runs(
        env_conf, policy, num_denoise, i_noise=i_noise
    )

    if env_conf.frozen_noise:
        rreturn_se = None

    return Trajectory(rreturn, None, None, rreturn_se), None


def evaluate_for_best(env_conf, policy, num_denoise_passiveuation):
    mean_ret, _, _ = mean_return_over_runs(
        env_conf, policy, num_denoise_passiveuation, i_noise=99999
    )
    return mean_ret
