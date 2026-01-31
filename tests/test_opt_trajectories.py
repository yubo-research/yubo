import numpy as np


class MockTrajectory:
    def __init__(self, rreturn):
        self.rreturn = rreturn


class MockEnvConf:
    def __init__(self):
        self.noise_seed_0 = 0
        self.frozen_noise = False


class MockPolicy:
    pass


def test_collect_trajectory_with_noise(monkeypatch):
    from optimizer import opt_trajectories

    def mock_collect_trajectory(env_conf, policy, noise_seed=0):
        return MockTrajectory(rreturn=1.5 + noise_seed * 0.01)

    monkeypatch.setattr(opt_trajectories, "collect_trajectory", mock_collect_trajectory)

    env_conf = MockEnvConf()
    policy = MockPolicy()

    traj, noise_seed = opt_trajectories.collect_trajectory_with_noise(env_conf, policy, i_noise=1, denoise_seed=0)
    assert traj.rreturn > 0
    assert noise_seed == 1


def test_mean_return_over_runs(monkeypatch):
    from optimizer import opt_trajectories

    call_count = [0]

    def mock_collect_trajectory(env_conf, policy, noise_seed=0):
        call_count[0] += 1
        return MockTrajectory(rreturn=1.0 + call_count[0] * 0.1)

    monkeypatch.setattr(opt_trajectories, "collect_trajectory", mock_collect_trajectory)

    env_conf = MockEnvConf()
    policy = MockPolicy()

    mean_ret, se_ret, all_same = opt_trajectories.mean_return_over_runs(env_conf, policy, num_denoise=3, i_noise=0)
    assert np.isfinite(mean_ret)
    assert np.isfinite(se_ret)


def test_collect_denoised_trajectory_single(monkeypatch):
    from optimizer import opt_trajectories

    def mock_collect_trajectory(env_conf, policy, noise_seed=0):
        return MockTrajectory(rreturn=2.0)

    monkeypatch.setattr(opt_trajectories, "collect_trajectory", mock_collect_trajectory)

    env_conf = MockEnvConf()
    policy = MockPolicy()

    traj, _ = opt_trajectories.collect_denoised_trajectory(env_conf, policy, num_denoise=1)
    assert traj.rreturn == 2.0


def test_collect_denoised_trajectory_multiple(monkeypatch):
    from optimizer import opt_trajectories

    counter = [0]

    def mock_collect_trajectory(env_conf, policy, noise_seed=0):
        counter[0] += 1
        return MockTrajectory(rreturn=1.0 + counter[0] * 0.1)

    monkeypatch.setattr(opt_trajectories, "collect_trajectory", mock_collect_trajectory)

    env_conf = MockEnvConf()
    policy = MockPolicy()

    traj, _ = opt_trajectories.collect_denoised_trajectory(env_conf, policy, num_denoise=3)
    assert np.isfinite(traj.rreturn)


def test_evaluate_for_best(monkeypatch):
    from optimizer import opt_trajectories

    def mock_collect_trajectory(env_conf, policy, noise_seed=0):
        return MockTrajectory(rreturn=5.0)

    monkeypatch.setattr(opt_trajectories, "collect_trajectory", mock_collect_trajectory)

    env_conf = MockEnvConf()
    policy = MockPolicy()

    ret = opt_trajectories.evaluate_for_best(env_conf, policy, num_denoise_passiveuation=2)
    assert ret == 5.0
